"""Structural Attention Bias (SAB) — independent extension on top of TableLoRA.

Design constraints (see HANDOFF.md §11 + the SAB task spec):

1. Must NOT modify any existing TableLoRA code (table_lora.py, load_table_lora,
   PeftModel_new.forward, Linear_new.forward, DataCollatorForSeq2Seq_new).
2. Must coexist with load_table_lora() — uses different attribute names and
   patches a different upstream symbol (`LlamaAttention.forward`).
3. Must default OFF. Removing this directory must leave TableLoRA 100%
   functional.
4. With `use_sab=True` but `alpha_row=alpha_col=0` the forward must produce
   logits that are bit-identical to `use_sab=False` (the non-invasiveness test).

What it does mathematically
---------------------------
For each attention layer, immediately before the softmax, we add a per-(layer,
head) bias to the pre-softmax logits:

    bias[b, h, i, j] = alpha_row[layer, h] * same_row[b, i, j]
                     + alpha_col[layer, h] * same_col[b, i, j]

where two positions (i, j) are in the "same row" iff they share row_id AND
both belong to the table region (row_id != 0 OR col_id != 0). same_col is
defined symmetrically. alpha_row / alpha_col are learnable scalars,
zero-initialised so the first training step is identical to the baseline.

How it is wired
---------------
* `install_sab_attention_patch()` monkey-patches
  `transformers.models.llama.modeling_llama.LlamaAttention.forward` with
  `_sab_llama_attention_forward`. This is a verbatim copy of the upstream
  forward (else branch only — DeepSeek has pretraining_tp=1 so the upstream
  if-branch never runs) plus a single block that adds the SAB bias right
  after the causal mask add and before softmax. When the SAB context is
  empty (no module installed, or no row/col masks cached), that block is a
  no-op and the function is mathematically and bit-equivalent to upstream.

* `load_sab(model)` walks down to the inner LlamaModel, builds an
  `SABModule(num_layers, num_heads)` and registers it as `model.sab_module`
  so its parameters are part of the parameter tree. It also registers a
  forward_pre_hook on `model` (with_kwargs=True) that pulls `row_ids` and
  `col_ids` out of the kwargs about to be passed to forward(), pre-computes
  the (same_row, same_col) boolean masks, and stashes them in `_SAB_CTX`.
  When the patched LlamaAttention.forward runs, every layer reads those
  masks from `_SAB_CTX` along with its own `self.layer_idx`.

What we deliberately do NOT touch
---------------------------------
* TableLoRA's setattr-on-Linear mechanism for row_ids/col_ids — we do not
  setattr anything onto attention modules. Communication is via the global
  `_SAB_CTX` dict, which TableLoRA does not read.
* PeftModel_new.forward / prepare_inputs_for_generation — they remain
  exactly as TableLoRA wrote them. The pre_hook fires from the PyTorch
  module hook system, before forward() is invoked.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# `module`           : the currently active SABModule (or None when SAB is off)
# `same_row`         : (bs, seq, seq) bool tensor for the current forward step
# `same_col`         : (bs, seq, seq) bool tensor for the current forward step
# `patched`               : whether attention forwards have been replaced
# `_orig_fwd`             : upstream LlamaAttention.forward (for restore)
# `_orig_sdpa_fwd`        : upstream LlamaSdpaAttention.forward (for restore)
# `_orig_qwen2_fwd`       : upstream Qwen2Attention.forward (for restore)
# `_orig_qwen2_sdpa_fwd`  : upstream Qwen2SdpaAttention.forward (for restore)
# `hook_count`            : forward_pre_hook fires (training / direct forward path)
# `fallback_count`        : generate-path fallback engaged (read row_ids off PEFT
#                           Linear children of the attention module). Mainly used
#                           by tests / smoke checks to verify both paths are wired.
_SAB_CTX: Dict[str, Any] = {
    "module": None,
    "same_row": None,
    "same_col": None,
    "patched": False,
    "_orig_fwd": None,
    "_orig_sdpa_fwd": None,
    "_orig_qwen2_fwd": None,
    "_orig_qwen2_sdpa_fwd": None,
    "hook_count": 0,
    "fallback_count": 0,
}


# ---------------------------------------------------------------------------
# SABModule
# ---------------------------------------------------------------------------
class SABModule(nn.Module):
    """Holds the learnable per-(layer, head) row/col attention bias scalars.

    Parameter count: num_layers * num_heads * 2.
    For DeepSeek-7B-Chat that is 30 * 32 * 2 = 1920 scalars.

    Both alpha_row and alpha_col are zero-initialised so that the first
    forward pass after attaching SAB produces exactly the same output as the
    baseline (this is the non-invasiveness contract).
    """

    def __init__(self, num_layers: int, num_heads: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.alpha_row = nn.Parameter(torch.zeros(self.num_layers, self.num_heads))
        self.alpha_col = nn.Parameter(torch.zeros(self.num_layers, self.num_heads))


# ---------------------------------------------------------------------------
# Mask computation + context plumbing
# ---------------------------------------------------------------------------
def _compute_same_masks(row_ids: torch.Tensor, col_ids: torch.Tensor):
    """Return (same_row, same_col) boolean tensors of shape (bs, seq, seq).

    A position is a "table token" iff (row_id != 0) OR (col_id != 0).
    Two positions are same_row iff they share row_id AND both are table tokens
    (and symmetrically for same_col). Diagonal entries are True (a token is
    trivially in its own row/col), but the diagonal contribution is harmless
    since it adds the same constant to every query position's self-attention.
    """
    if row_ids.dim() != 2 or col_ids.dim() != 2:
        raise ValueError(
            f"row_ids/col_ids must be (bs, seq), got {tuple(row_ids.shape)} / {tuple(col_ids.shape)}"
        )
    is_table = (row_ids != 0) | (col_ids != 0)  # (bs, seq) bool
    # Outer-equality + table-mask gating, both axes
    same_row = (
        (row_ids[:, :, None] == row_ids[:, None, :])
        & is_table[:, :, None]
        & is_table[:, None, :]
    )
    same_col = (
        (col_ids[:, :, None] == col_ids[:, None, :])
        & is_table[:, :, None]
        & is_table[:, None, :]
    )
    return same_row, same_col


def set_sab_inputs(row_ids: Optional[torch.Tensor], col_ids: Optional[torch.Tensor]) -> None:
    """Stash the per-step (same_row, same_col) masks for the patched attention
    forward to read. Pass row_ids=col_ids=None to clear (i.e. SAB no-op for
    this step)."""
    if row_ids is None or col_ids is None:
        _SAB_CTX["same_row"] = None
        _SAB_CTX["same_col"] = None
        return
    same_row, same_col = _compute_same_masks(row_ids, col_ids)
    _SAB_CTX["same_row"] = same_row
    _SAB_CTX["same_col"] = same_col


def _try_compute_masks_from_attn(
    attn_module, q_len: int, k_len: int
):
    """Generate-path fallback. The forward_pre_hook on PeftModelForCausalLM
    only fires for `model(...)` calls. `model.generate(...)` instead goes
    `generate -> base_model.generate -> prepare_inputs_for_generation ->
    base_model.forward`, never invoking the hook on the peft wrapper.

    But TableLoRA's `_table_lora_set_row_col_ids_on_modules` always pushes
    row_ids/col_ids onto every PEFT `Linear` (k_proj/v_proj for the default
    `lora_target=k_proj,v_proj`) BEFORE every base_model forward, in BOTH
    training and inference paths. So we can read row_ids/col_ids off any
    PEFT Linear child of this attention module and re-derive the masks
    locally — no hook needed, no TableLoRA code modified.

    We only return masks when `q_len == k_len == row_ids.shape[-1]`, i.e.
    training step or generate prefill. Decode steps (q_len=1, k_len=cache)
    are intentionally skipped: the new query token is always a response
    token (row=col=0, not a table token), so its SAB bias to any key would
    be zero anyway, and TableLoRA only stores per-step row_ids of width 1
    in that case (no full-history mask available).
    """
    if q_len != k_len:
        return None, None
    for child_name in ("k_proj", "v_proj", "q_proj", "o_proj"):
        child = getattr(attn_module, child_name, None)
        if child is None:
            continue
        rid = getattr(child, "row_ids", None)
        cid = getattr(child, "col_ids", None)
        if rid is None or cid is None:
            continue
        if not torch.is_tensor(rid) or not torch.is_tensor(cid):
            continue
        if rid.dim() != 2 or cid.dim() != 2:
            continue
        if rid.shape[-1] != q_len:
            continue
        return _compute_same_masks(rid, cid)
    return None, None


def clear_sab_inputs() -> None:
    _SAB_CTX["same_row"] = None
    _SAB_CTX["same_col"] = None


# ---------------------------------------------------------------------------
# Patched LlamaAttention.forward
# ---------------------------------------------------------------------------
# Verbatim copy of transformers 4.46.1 LlamaAttention.forward (else branch of
# pretraining_tp; DeepSeek has tp=1 so the if branch is never taken). The only
# addition is the SAB BIAS INJECTION block, which is a no-op when _SAB_CTX has
# no module / no cached masks.
#
# We import the helpers (apply_rotary_pos_emb, repeat_kv) lazily inside the
# function so that simply importing this file does not pull in the transformers
# llama module — that way `from llamafactory.sab import ...` is cheap and safe
# even when transformers is mocked.
def _sab_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    **kwargs,
):
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.size()

    # NOTE: skipping the upstream `pretraining_tp > 1` branch. Llama/DeepSeek
    # default to pretraining_tp=1 so this is the path the upstream forward
    # always takes for our model — mathematically (and bit-) identical.
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ===== SAB BIAS INJECTION (only addition vs upstream) =====
    sab_module = _SAB_CTX["module"]
    if sab_module is not None:
        same_row = _SAB_CTX["same_row"]
        same_col = _SAB_CTX["same_col"]
        # Generate-path fallback. If the forward_pre_hook on the peft model
        # didn't fire (because we're inside model.generate(), which bypasses
        # model.__call__), try to read the masks straight off TableLoRA's
        # per-Linear row_ids/col_ids stash. See _try_compute_masks_from_attn.
        if same_row is None or same_col is None:
            same_row, same_col = _try_compute_masks_from_attn(
                self, q_len, key_states.shape[-2]
            )
            if same_row is not None:
                _SAB_CTX["fallback_count"] += 1

        if same_row is not None and same_col is not None:
            layer_idx = self.layer_idx
            # alpha_*[layer_idx] -> shape (num_heads,) -> (1, h, 1, 1)
            ar = sab_module.alpha_row[layer_idx].to(attn_weights.dtype).view(1, -1, 1, 1)
            ac = sab_module.alpha_col[layer_idx].to(attn_weights.dtype).view(1, -1, 1, 1)
            # same_row/col are (bs, seq_q_full, seq_k_full) bool. The Q/K dims
            # may be smaller than seq_full at decoding time (1-token decode step
            # or KV-cache shorter than the cached masks); slice both axes.
            sr = same_row.to(attn_weights.dtype).unsqueeze(1)  # (bs, 1, seq_q_full, seq_k_full)
            sc = same_col.to(attn_weights.dtype).unsqueeze(1)
            sr = sr[..., -q_len:, : key_states.shape[-2]]
            sc = sc[..., -q_len:, : key_states.shape[-2]]
            sab_bias = ar * sr + ac * sc
            attn_weights = attn_weights + sab_bias
    # ===== END SAB INJECTION =====

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Patched LlamaSdpaAttention.forward (SDPA path — preferred)
# ---------------------------------------------------------------------------
# Instead of forcing eager attention and materializing the full (bs, h, seq,
# seq) attention matrix, we inject the SAB bias directly into the `causal_mask`
# passed to `torch.nn.functional.scaled_dot_product_attention`. SDPA's fused
# kernel handles the combined mask internally, so we get SAB + SDPA speed.
#
# This is a near-verbatim copy of transformers 4.46.1 LlamaSdpaAttention.forward
# with only the SAB BIAS INJECTION block added right before the SDPA call.

def _sab_llama_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    **kwargs,
):
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    # When output_attentions=True, SDPA falls back to eager (super().forward).
    # Our eager patch (_sab_llama_attention_forward) handles that case, so just
    # delegate to it directly.
    if output_attentions:
        return _sab_llama_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # ===== SAB BIAS INJECTION (SDPA path) =====
    # Add SAB bias directly into the causal_mask so SDPA's fused kernel
    # handles both the causal mask and the structural bias in one pass.
    sab_module = _SAB_CTX["module"]
    if sab_module is not None:
        same_row = _SAB_CTX["same_row"]
        same_col = _SAB_CTX["same_col"]
        if same_row is None or same_col is None:
            same_row, same_col = _try_compute_masks_from_attn(
                self, q_len, key_states.shape[-2]
            )
            if same_row is not None:
                _SAB_CTX["fallback_count"] += 1

        if same_row is not None and same_col is not None:
            layer_idx = self.layer_idx
            _dtype = causal_mask.dtype if causal_mask is not None else query_states.dtype
            ar = sab_module.alpha_row[layer_idx].to(_dtype).view(1, -1, 1, 1)
            ac = sab_module.alpha_col[layer_idx].to(_dtype).view(1, -1, 1, 1)
            sr = same_row.to(_dtype).unsqueeze(1)
            sc = same_col.to(_dtype).unsqueeze(1)
            k_len = key_states.shape[-2]
            sr = sr[..., -q_len:, :k_len]
            sc = sc[..., -q_len:, :k_len]
            sab_bias = ar * sr + ac * sc
            if causal_mask is not None:
                causal_mask = causal_mask + sab_bias
            else:
                # SDPA would normally use is_causal=True (no explicit mask).
                # We need an explicit causal mask to combine with our bias,
                # otherwise is_causal would be set to False and the model
                # loses causal masking entirely.
                min_val = torch.finfo(_dtype).min
                cm = torch.full(
                    (1, 1, q_len, k_len), 0.0,
                    device=query_states.device, dtype=_dtype,
                )
                cm.masked_fill_(
                    torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=query_states.device),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0),
                    min_val,
                )
                causal_mask = cm + sab_bias
    # ===== END SAB INJECTION =====

    # SDPA contiguity fix (torch==2.1.2 bug workaround from upstream)
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# ---------------------------------------------------------------------------
# Patched Qwen2Attention.forward (eager path)
# ---------------------------------------------------------------------------
# Near-verbatim copy of transformers 4.46.x Qwen2Attention.forward with the
# same SAB BIAS INJECTION block. Qwen2's attention is structurally identical
# to LLaMA's — same RoPE, same repeat_kv, same softmax path — so the
# injection point and the bias arithmetic are unchanged.
# Importing from qwen2 lazily so this file is safe to import on LLaMA-only runs.
def _sab_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    **kwargs,
):
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ===== SAB BIAS INJECTION (only addition vs upstream) =====
    sab_module = _SAB_CTX["module"]
    if sab_module is not None:
        same_row = _SAB_CTX["same_row"]
        same_col = _SAB_CTX["same_col"]
        if same_row is None or same_col is None:
            same_row, same_col = _try_compute_masks_from_attn(
                self, q_len, key_states.shape[-2]
            )
            if same_row is not None:
                _SAB_CTX["fallback_count"] += 1

        if same_row is not None and same_col is not None:
            layer_idx = self.layer_idx
            ar = sab_module.alpha_row[layer_idx].to(attn_weights.dtype).view(1, -1, 1, 1)
            ac = sab_module.alpha_col[layer_idx].to(attn_weights.dtype).view(1, -1, 1, 1)
            sr = same_row.to(attn_weights.dtype).unsqueeze(1)
            sc = same_col.to(attn_weights.dtype).unsqueeze(1)
            sr = sr[..., -q_len:, : key_states.shape[-2]]
            sc = sc[..., -q_len:, : key_states.shape[-2]]
            sab_bias = ar * sr + ac * sc
            attn_weights = attn_weights + sab_bias
    # ===== END SAB INJECTION =====

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Patched Qwen2SdpaAttention.forward (SDPA path — preferred)
# ---------------------------------------------------------------------------
def _sab_qwen2_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    **kwargs,
):
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    if output_attentions:
        return _sab_qwen2_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # ===== SAB BIAS INJECTION (SDPA path) =====
    sab_module = _SAB_CTX["module"]
    if sab_module is not None:
        same_row = _SAB_CTX["same_row"]
        same_col = _SAB_CTX["same_col"]
        if same_row is None or same_col is None:
            same_row, same_col = _try_compute_masks_from_attn(
                self, q_len, key_states.shape[-2]
            )
            if same_row is not None:
                _SAB_CTX["fallback_count"] += 1

        if same_row is not None and same_col is not None:
            layer_idx = self.layer_idx
            _dtype = causal_mask.dtype if causal_mask is not None else query_states.dtype
            ar = sab_module.alpha_row[layer_idx].to(_dtype).view(1, -1, 1, 1)
            ac = sab_module.alpha_col[layer_idx].to(_dtype).view(1, -1, 1, 1)
            sr = same_row.to(_dtype).unsqueeze(1)
            sc = same_col.to(_dtype).unsqueeze(1)
            k_len = key_states.shape[-2]
            sr = sr[..., -q_len:, :k_len]
            sc = sc[..., -q_len:, :k_len]
            sab_bias = ar * sr + ac * sc
            if causal_mask is not None:
                causal_mask = causal_mask + sab_bias
            else:
                min_val = torch.finfo(_dtype).min
                cm = torch.full(
                    (1, 1, q_len, k_len), 0.0,
                    device=query_states.device, dtype=_dtype,
                )
                cm.masked_fill_(
                    torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=query_states.device),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0),
                    min_val,
                )
                causal_mask = cm + sab_bias
    # ===== END SAB INJECTION =====

    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# ---------------------------------------------------------------------------
# Patch install / uninstall
# ---------------------------------------------------------------------------
def install_sab_attention_patch() -> None:
    """Patch both LLaMA and Qwen2 attention classes with SAB-aware forwards.

    Patches four classes in total:
      - LlamaAttention.forward       (eager, used by DeepSeek / LLaMA models)
      - LlamaSdpaAttention.forward   (SDPA, preferred by DeepSeek / LLaMA)
      - Qwen2Attention.forward       (eager, used by Qwen2 models)
      - Qwen2SdpaAttention.forward   (SDPA, preferred by Qwen2)

    Patching both families is safe because only the class actually instantiated
    by the loaded model will ever be called — patching the other family is a
    no-op at runtime. DeepSeek runs are therefore fully unaffected by the Qwen2
    patches and vice-versa. Idempotent: calling a second time is a no-op.
    """
    if _SAB_CTX["patched"]:
        return

    import transformers.models.llama.modeling_llama as llm_mod
    _SAB_CTX["_orig_fwd"] = llm_mod.LlamaAttention.forward
    _SAB_CTX["_orig_sdpa_fwd"] = llm_mod.LlamaSdpaAttention.forward
    llm_mod.LlamaAttention.forward = _sab_llama_attention_forward
    llm_mod.LlamaSdpaAttention.forward = _sab_llama_sdpa_attention_forward

    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        _SAB_CTX["_orig_qwen2_fwd"] = qwen2_mod.Qwen2Attention.forward
        _SAB_CTX["_orig_qwen2_sdpa_fwd"] = qwen2_mod.Qwen2SdpaAttention.forward
        qwen2_mod.Qwen2Attention.forward = _sab_qwen2_attention_forward
        qwen2_mod.Qwen2SdpaAttention.forward = _sab_qwen2_sdpa_attention_forward
        print("[sab] Qwen2Attention.forward + Qwen2SdpaAttention.forward monkey-patched.")
    except (ImportError, AttributeError):
        # Older transformers without Qwen2 support — silently skip.
        pass

    _SAB_CTX["patched"] = True
    print("[sab] LlamaAttention.forward + LlamaSdpaAttention.forward monkey-patched.")


def uninstall_sab_attention_patch() -> None:
    """Restore original forwards. Mainly for unit tests."""
    if not _SAB_CTX["patched"]:
        return

    import transformers.models.llama.modeling_llama as llm_mod
    if _SAB_CTX.get("_orig_fwd"):
        llm_mod.LlamaAttention.forward = _SAB_CTX["_orig_fwd"]
    if _SAB_CTX.get("_orig_sdpa_fwd"):
        llm_mod.LlamaSdpaAttention.forward = _SAB_CTX["_orig_sdpa_fwd"]
    _SAB_CTX["_orig_fwd"] = None
    _SAB_CTX["_orig_sdpa_fwd"] = None

    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        if _SAB_CTX.get("_orig_qwen2_fwd"):
            qwen2_mod.Qwen2Attention.forward = _SAB_CTX["_orig_qwen2_fwd"]
        if _SAB_CTX.get("_orig_qwen2_sdpa_fwd"):
            qwen2_mod.Qwen2SdpaAttention.forward = _SAB_CTX["_orig_qwen2_sdpa_fwd"]
    except (ImportError, AttributeError):
        pass
    _SAB_CTX["_orig_qwen2_fwd"] = None
    _SAB_CTX["_orig_qwen2_sdpa_fwd"] = None

    _SAB_CTX["patched"] = False
    print("[sab] LlamaAttention + LlamaSdpaAttention + Qwen2Attention + Qwen2SdpaAttention forwards restored.")


# ---------------------------------------------------------------------------
# Model attachment helpers
# ---------------------------------------------------------------------------
def _find_inner_llama(model: nn.Module) -> nn.Module:
    """Walk down PEFT/HF wrappers to find the innermost LlamaModel.

    The chain is typically:
        PeftModelForCausalLM
        └─ base_model: LoraModel
           └─ model: LlamaForCausalLM
              └─ model: LlamaModel        <- the one with `.layers`
    """
    visited = set()
    cur = model
    for _ in range(10):  # bounded walk
        if id(cur) in visited:
            break
        visited.add(id(cur))
        # LlamaModel exposes `.layers` (a ModuleList of LlamaDecoderLayer)
        if hasattr(cur, "layers") and isinstance(cur.layers, nn.ModuleList):
            return cur
        if hasattr(cur, "model") and isinstance(cur.model, nn.Module):
            cur = cur.model
            continue
        if hasattr(cur, "base_model") and isinstance(cur.base_model, nn.Module):
            cur = cur.base_model
            continue
        break
    raise RuntimeError(
        "load_sab: could not locate the inner LlamaModel under the wrapper chain "
        "(no .layers ModuleList found)."
    )


def _build_pre_hook():
    def _sab_pre_forward_hook(module, args, kwargs):
        # Pull row_ids/col_ids straight out of the kwargs about to be sent to
        # forward(). PeftModel_new.forward declares both as keyword args.
        row_ids = kwargs.get("row_ids", None)
        col_ids = kwargs.get("col_ids", None)
        set_sab_inputs(row_ids, col_ids)
        if row_ids is not None and col_ids is not None:
            _SAB_CTX["hook_count"] += 1
        return None  # do not modify args/kwargs

    return _sab_pre_forward_hook


def _build_post_hook():
    def _sab_post_forward_hook(module, input, output):
        # Clear the cached masks after every model.forward() call.
        # Without this, _SAB_CTX["same_row"/"same_col"] would persist into a
        # subsequent model.generate() call (which never fires the pre_hook),
        # and the patched LlamaAttention.forward would silently consume stale
        # masks from the previous forward instead of falling back to reading
        # row_ids/col_ids off the per-layer PEFT Linear children.
        # See sab_generate_test.py for the regression that caught this.
        clear_sab_inputs()
        return None

    return _sab_post_forward_hook


def load_sab(
    model: nn.Module,
    num_heads: Optional[int] = None,
) -> SABModule:
    """Install SAB on a (peft-wrapped) model.

    Steps:
      1. Walk down to the inner LlamaModel.
      2. Verify the model is using eager attention (or warn loudly).
      3. Create an SABModule with the right (num_layers, num_heads) shape.
      4. Register it as `model.sab_module` so its parameters live in the
         parameter tree (so the optimizer / `.parameters()` picks them up).
      5. Install the LlamaAttention.forward monkey-patch (idempotent).
      6. Register a forward_pre_hook on `model` that pulls row_ids/col_ids
         from forward kwargs and stashes the (same_row, same_col) masks.

    Returns the freshly attached SABModule. Idempotent: calling load_sab on a
    model that already has `sab_module` will replace it.
    """
    inner = _find_inner_llama(model)
    num_layers = len(inner.layers)
    if num_heads is None:
        # All Llama layers share num_heads
        num_heads = inner.layers[0].self_attn.num_heads

    # Info: print which attention implementation is in use. Both eager and
    # SDPA are now supported (we patch both classes in install_sab_attention_patch).
    cfg = getattr(inner, "config", None)
    impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    if impl == "flash_attention_2":
        print(
            "[sab] WARNING: model uses flash_attention_2. SAB does NOT patch "
            "LlamaFlashAttention2 — the bias will silently NOT be applied. "
            "Use sdpa or eager instead.",
            flush=True,
        )
    else:
        print(f"[sab] attn_implementation={impl!r} — OK (eager and sdpa both supported).", flush=True)

    # Build the SABModule on the same device as the rest of the model
    sample_param = next(inner.parameters())
    sab_module = SABModule(num_layers=num_layers, num_heads=num_heads)
    sab_module = sab_module.to(device=sample_param.device, dtype=torch.float32)

    # Attach as a registered submodule. If a previous SAB lives here, drop it.
    if hasattr(model, "sab_module"):
        try:
            del model._modules["sab_module"]
        except (KeyError, AttributeError):
            pass
    model.add_module("sab_module", sab_module)

    install_sab_attention_patch()
    _SAB_CTX["module"] = sab_module

    # Register the pre-forward hook (sets the cached masks from kwargs) AND
    # a post-forward hook (clears the cache so it doesn't leak into a later
    # generate() call where the hook can't fire). PeftModel forward accepts
    # row_ids/col_ids as keyword args; the pre-hook with with_kwargs=True
    # will see them. Both handles are stored on the model so unload_sab can
    # remove them.
    for attr in ("_sab_pre_hook_handle", "_sab_post_hook_handle", "_sab_hook_handle"):
        if hasattr(model, attr):
            try:
                getattr(model, attr).remove()
            except Exception:
                pass
            try:
                delattr(model, attr)
            except Exception:
                pass
    model._sab_pre_hook_handle = model.register_forward_pre_hook(
        _build_pre_hook(), with_kwargs=True
    )
    model._sab_post_hook_handle = model.register_forward_hook(_build_post_hook())

    n_params = sab_module.alpha_row.numel() + sab_module.alpha_col.numel()
    print(
        f"[sab] load_sab OK. num_layers={num_layers} num_heads={num_heads} "
        f"trainable_params={n_params} attach_point={type(model).__name__}.sab_module"
    )
    return sab_module


# ---------------------------------------------------------------------------
# Save / load SAB state
# ---------------------------------------------------------------------------
# PEFT's save_pretrained only writes the registered adapter weights (LoRA +
# prompt_encoder). Random submodules attached to the peft model after wrapping
# — like our `sab_module` — are silently dropped. So we save SAB ourselves
# alongside the adapter, and reload it inside the load_model wrapper used by
# the LF inference path. See sab_save_load_test.py for the verification.

SAB_STATE_FILENAME = "sab_module.safetensors"


def save_sab(model: nn.Module, save_dir: str) -> Optional[str]:
    """Write the SAB state dict (alpha_row, alpha_col) to
    `<save_dir>/sab_module.safetensors`. Returns the path written, or None
    if the model has no `sab_module` attached (so the call is a no-op when
    SAB is disabled — safe to register as a TrainerCallback unconditionally).
    """
    sab_module = getattr(model, "sab_module", None)
    if sab_module is None:
        return None
    import os as _os
    from safetensors.torch import save_file as _st_save

    _os.makedirs(save_dir, exist_ok=True)
    state = {k: v.detach().to(torch.float32).cpu() for k, v in sab_module.state_dict().items()}
    out_path = _os.path.join(save_dir, SAB_STATE_FILENAME)
    _st_save(state, out_path)
    print(f"[sab] save_sab wrote {len(state)} tensors to {out_path}", flush=True)
    return out_path


def load_sab_state(model: nn.Module, load_dir: str, strict: bool = True) -> bool:
    """Look for `<load_dir>/sab_module.safetensors`; if present, load it into
    `model.sab_module`. Returns True iff state was loaded.

    Used by the LF inference path: after `load_sab(model)` creates a fresh
    zero-initialised SABModule, this restores the trained alpha values from
    disk so eval reproduces what training learnt.
    """
    sab_module = getattr(model, "sab_module", None)
    if sab_module is None:
        if strict:
            raise RuntimeError("load_sab_state called but model has no sab_module")
        return False
    import os as _os
    from safetensors.torch import load_file as _st_load

    state_path = _os.path.join(load_dir, SAB_STATE_FILENAME)
    if not _os.path.exists(state_path):
        if strict:
            print(f"[sab] load_sab_state: no {SAB_STATE_FILENAME} in {load_dir}", flush=True)
        return False
    sample_p = next(sab_module.parameters())
    state = _st_load(state_path, device=str(sample_p.device))
    # Cast to the param dtype (we always store fp32, but params may be fp32 too)
    state = {k: v.to(sample_p.dtype) for k, v in state.items()}
    missing, unexpected = sab_module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[sab] load_sab_state: missing={missing} unexpected={unexpected}",
            flush=True,
        )
    print(
        f"[sab] load_sab_state: loaded {len(state)} tensors from {state_path}; "
        f"alpha_row.absmax={sab_module.alpha_row.abs().max().item():.4e} "
        f"alpha_col.absmax={sab_module.alpha_col.abs().max().item():.4e}",
        flush=True,
    )
    return True


def unload_sab(model: nn.Module) -> None:
    """Tear down SAB: remove pre/post hooks, drop the SABModule, restore upstream
    LlamaAttention.forward, and clear the global context. Mainly for tests."""
    for attr in ("_sab_pre_hook_handle", "_sab_post_hook_handle", "_sab_hook_handle"):
        if hasattr(model, attr):
            try:
                getattr(model, attr).remove()
            except Exception:
                pass
            try:
                delattr(model, attr)
            except Exception:
                pass
    if hasattr(model, "sab_module"):
        try:
            del model._modules["sab_module"]
        except (KeyError, AttributeError):
            pass
    _SAB_CTX["module"] = None
    clear_sab_inputs()
    uninstall_sab_attention_patch()
