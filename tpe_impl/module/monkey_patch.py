"""Monkey-patch TableLoRA's `Linear_new.forward` to inject tpe path signal.

Strategy: replace the forward entirely (not wrap), because the original
forward has no `pos_signal_K / pos_signal_V` intermediate — the 2D LoRA
contribution is the inline expression
    `lora_B_tab(col_embeds + row_embeds) * scaling_tab`
at Linear_new.forward:1206. Our patch rewrites that line to
    `lora_B_tab(col_embeds + row_embeds + top_pooled + left_pooled) * scaling_tab`
with `top_pooled / left_pooled = 0` when:
  - use_tpe disabled (patch isn't installed)
  - `_TPE_CTX["top_path_ids"] is None` (hook didn't fire, e.g. inference without kwargs)
  - embeddings are zero-init (pre-training)

Any of those three → bit-exact equivalence with TableLoRA.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

# Module-level context, populated by outer model forward_pre_hook, cleared by post_hook.
_TPE_CTX: dict = {"top_path_ids": None, "left_path_ids": None}
# The per-layer TPEPathModuleList. Set by install_tpe_patch().
_TPE_MODULES: Optional[torch.nn.Module] = None
# Whether the patch is installed (idempotency flag).
_PATCH_INSTALLED: bool = False
# Saved reference to the original Linear_new.forward (before our patch).
_ORIG_LINEAR_FORWARD = None
# Ablation flag: when True, drop col_embeds + row_embeds from pos_signal so only
# top_pooled + left_pooled feeds lora_B_tab. Set by install_tpe_patch().
_TPE_DISABLE_2D_LORA: bool = False


def _patched_linear_new_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Drop-in replacement for Linear_new.forward.

    Identical to upstream except `lora_B_tab(col_embeds + row_embeds)` becomes
    `lora_B_tab(col_embeds + row_embeds + top_pooled + left_pooled)`.
    """
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if adapter_names is not None:
        return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_cast = x.to(lora_A.weight.dtype)

        if self.is_tab_lora:
            lora_A_tab_col = self.lora_A_tab_col[active_adapter]
            lora_A_tab_row = self.lora_A_tab_row[active_adapter]
            lora_B_tab = self.lora_B_tab[active_adapter]
            scaling_tab = self.scaling_tab[active_adapter]

            # 2D LoRA row/col contribution. Ablation: when _TPE_DISABLE_2D_LORA
            # is True, drop col_embeds + row_embeds entirely so pos_signal is
            # built only from the tpe branch.
            if not _TPE_DISABLE_2D_LORA:
                col_embeds = lora_A_tab_col(self.col_ids)
                row_embeds = lora_A_tab_row(self.row_ids)
                pos_signal = col_embeds + row_embeds
            else:
                pos_signal = None  # set by tpe branch below, or term skipped

            # TPE additions. Guard with all three fail-soft conditions:
            # 1. TPE_MODULES set (patch was installed)
            # 2. This Linear_new was bound to a layer_idx (k_proj or v_proj)
            # 3. _TPE_CTX has path_ids (forward_pre_hook fired this forward)
            tpe_layer_idx = getattr(self, "_tpe_layer_idx", None)
            if (_TPE_MODULES is not None
                    and tpe_layer_idx is not None
                    and _TPE_CTX.get("top_path_ids") is not None):
                tpe_module = _TPE_MODULES[tpe_layer_idx]
                is_k_proj = getattr(self, "_tpe_is_k_proj", False)
                emb_top = tpe_module.emb_top_path_K if is_k_proj else tpe_module.emb_top_path_V
                emb_left = tpe_module.emb_left_path_K if is_k_proj else tpe_module.emb_left_path_V

                top_pooled = tpe_module.pool(_TPE_CTX["top_path_ids"], emb_top)
                left_pooled = tpe_module.pool(_TPE_CTX["left_path_ids"], emb_left)
                if pos_signal is not None:
                    top_pooled = top_pooled.to(pos_signal.dtype)
                    left_pooled = left_pooled.to(pos_signal.dtype)
                    pos_signal = pos_signal + top_pooled + left_pooled
                else:
                    # Ablation path: align tpe dtype to lora_B_tab weight dtype
                    target_dtype = lora_B_tab.weight.dtype
                    top_pooled = top_pooled.to(target_dtype)
                    left_pooled = left_pooled.to(target_dtype)
                    pos_signal = top_pooled + left_pooled

            result = result + lora_B(lora_A(dropout(x_cast))) * scaling
            if pos_signal is not None:
                result = result + lora_B_tab(pos_signal) * scaling_tab
        else:
            # Non-table LoRA branch: identical to upstream
            result = result + lora_B(lora_A(dropout(x_cast))) * scaling

    return result.to(torch_result_dtype)


# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

def _walk_linear_new_modules(model):
    """Yield (layer_idx, is_k_proj, module, name) for every LoRA-wrapped Linear
    that sits on a per-layer k_proj / v_proj.

    NOTE: after load_table_lora(), peft's Linear has its .__init__ and .forward
    replaced by Linear_new's, but the CLASS itself is still the peft Linear.
    So we check isinstance against the peft class, not the TableLoRA class name.
    """
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    for name, module in model.named_modules():
        if not isinstance(module, PeftLoraLinear):
            continue
        is_k = ".self_attn.k_proj" in name
        is_v = ".self_attn.v_proj" in name
        if not (is_k or is_v):
            continue
        # name example: "base_model.base_model.model.layers.5.self_attn.k_proj"
        try:
            idx_str = name.split(".layers.")[1].split(".")[0]
            layer_idx = int(idx_str)
        except (IndexError, ValueError):
            # Can't parse — skip (shouldn't normally happen with k/v_proj)
            continue
        yield layer_idx, is_k, module, name


def _build_pre_hook():
    def pre_hook(module, args, kwargs):
        # IMPORTANT: always overwrite (even with None) so stale ids from a prior
        # forward don't leak into an eval/generate call that has no path_ids.
        # Pop tpe kwargs so downstream transformer.forward doesn't choke on them.
        _TPE_CTX["top_path_ids"] = kwargs.pop("top_path_ids", None)
        _TPE_CTX["left_path_ids"] = kwargs.pop("left_path_ids", None)
        return args, kwargs
    return pre_hook


def _build_post_hook():
    def post_hook(module, args, output):
        # IMPORTANT: do NOT clear _TPE_CTX here. With gradient checkpointing,
        # the backward pass re-executes forward using the same outer-forward
        # invocation; the outer pre-hook does NOT re-fire for those inner
        # re-executions, so if we clear after the original forward the
        # re-forward sees _TPE_CTX=None, skips the tpe branch, and autograd
        # never connects tpe params into the graph (→ grads = None, weights
        # stay zero). Stale-value leakage is prevented by the pre-hook, which
        # now unconditionally overwrites on every outer forward (including with
        # None when the caller doesn't pass path_ids).
        return output
    return post_hook


def install_tpe_patch(model, tpe_config) -> None:
    """Install everything needed for use_tpe to take effect:

    1. Build `TPEPathModuleList(num_layers, vocab_size, low_rank_dim)`.
    2. Attach it to model as submodule `tpe_path_module` so params are picked up.
    3. Walk named_modules, tag each k_proj/v_proj Linear_new with
       `_tpe_layer_idx` + `_tpe_is_k_proj`.
    4. Register forward_pre_hook(with_kwargs=True) + forward_hook on model.
    5. Replace `Linear_new.forward` with our patched version (global, idempotent).

    If `tpe_config.use_tpe` is False, this is a no-op.
    """
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    global _TPE_MODULES, _PATCH_INSTALLED, _ORIG_LINEAR_FORWARD, _TPE_DISABLE_2D_LORA

    if not tpe_config.use_tpe:
        print("[tpe] install_tpe_patch: use_tpe=False, no-op.", flush=True)
        return

    # Set ablation flag from config (cheap; idempotent across re-installs).
    _TPE_DISABLE_2D_LORA = bool(getattr(tpe_config, "disable_2d_lora", False))
    if _TPE_DISABLE_2D_LORA:
        print("[tpe] install_tpe_patch: disable_2d_lora=True — pos_signal will "
              "EXCLUDE col_embeds + row_embeds (ablation mode).", flush=True)

    if _PATCH_INSTALLED:
        print("[tpe] install_tpe_patch: already installed, skipping.", flush=True)
        return

    # 1. Build module list
    from .tpe_path_module import TPEPathModuleList
    vocab_size = tpe_config.vocab_size
    dim = tpe_config.path_low_rank_dim
    inner = _find_inner_llama(model)
    num_layers = len(inner.layers)
    # When disable_2d_lora=True, tpe embeddings must non-zero-init or both
    # lora_B_tab and the embeddings remain stuck at zero (see ablation report).
    # Otherwise keep zero-init so use_tpe-untrained ≡ TableLoRA bit-exact.
    init_std = 0.02 if _TPE_DISABLE_2D_LORA else 0.0
    if init_std > 0:
        print(f"[tpe] non-zero-init tpe embeddings (std={init_std}) to break "
              "dead-init under disable_2d_lora=True.", flush=True)
    pool_mode = getattr(tpe_config, "pool_mode", "mean")
    top_only = bool(getattr(tpe_config, "tpe_top_only", False))
    if pool_mode != "mean" or top_only:
        print(f"[tpe] install_tpe_patch: pool_mode={pool_mode!r} "
              f"top_only={top_only}.", flush=True)
    module_list = TPEPathModuleList(
        num_layers=num_layers,
        path_vocab_size=vocab_size,
        low_rank_dim=dim,
        init_std=init_std,
        pool_mode=pool_mode,
        top_only=top_only,
    )
    sample_param = next(inner.parameters())
    module_list = module_list.to(device=sample_param.device, dtype=torch.float32)

    # 2. Attach as submodule
    if hasattr(model, "tpe_path_module"):
        try:
            del model._modules["tpe_path_module"]
        except (KeyError, AttributeError):
            pass
    model.add_module("tpe_path_module", module_list)
    _TPE_MODULES = module_list

    # 3. Tag each LoRA-wrapped k_proj / v_proj with layer idx + which projection
    tagged = 0
    for layer_idx, is_k, module, name in _walk_linear_new_modules(model):
        module._tpe_layer_idx = layer_idx
        module._tpe_is_k_proj = is_k
        tagged += 1

    # 4. Hook up pre/post hooks on OUTER model
    for attr in ("_tpe_pre_hook_handle", "_tpe_post_hook_handle"):
        if hasattr(model, attr):
            try:
                getattr(model, attr).remove()
            except Exception:
                pass
            try:
                delattr(model, attr)
            except Exception:
                pass
    model._tpe_pre_hook_handle = model.register_forward_pre_hook(
        _build_pre_hook(), with_kwargs=True
    )
    model._tpe_post_hook_handle = model.register_forward_hook(_build_post_hook())

    # 5. Replace PeftLoraLinear.forward (which is already TableLoRA-patched to
    # Linear_new.forward at this point) with our tpe-aware version.
    _ORIG_LINEAR_FORWARD = PeftLoraLinear.forward
    PeftLoraLinear.forward = _patched_linear_new_forward
    _PATCH_INSTALLED = True

    print(
        f"[tpe] install_tpe_patch OK. "
        f"num_layers={num_layers} vocab_size={vocab_size} low_rank_dim={dim} "
        f"tagged_linear_new={tagged}", flush=True,
    )


def uninstall_tpe_patch(model=None):
    """Revert the monkey-patch. Mostly for tests / sanity comparisons."""
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    global _TPE_MODULES, _PATCH_INSTALLED, _ORIG_LINEAR_FORWARD

    if not _PATCH_INSTALLED:
        return
    if _ORIG_LINEAR_FORWARD is not None:
        PeftLoraLinear.forward = _ORIG_LINEAR_FORWARD
    _ORIG_LINEAR_FORWARD = None
    _PATCH_INSTALLED = False
    _TPE_CTX["top_path_ids"] = None
    _TPE_CTX["left_path_ids"] = None
    if model is not None:
        for attr in ("_tpe_pre_hook_handle", "_tpe_post_hook_handle"):
            if hasattr(model, attr):
                try:
                    getattr(model, attr).remove()
                except Exception:
                    pass
                try:
                    delattr(model, attr)
                except Exception:
                    pass
        if hasattr(model, "tpe_path_module"):
            try:
                del model._modules["tpe_path_module"]
            except (KeyError, AttributeError):
                pass
    _TPE_MODULES = None
    print("[tpe] uninstall_tpe_patch done.", flush=True)


def _find_inner_llama(model):
    """Walk down to the inner LlamaModel (the module with `.layers`).

    Typical structure:
      PeftModel → .base_model → LoraModel → .model → LlamaForCausalLM → .model → LlamaModel

    Some wrappers may have self-referential `.base_model`, so we descend via
    a bounded traversal using an identity set to avoid infinite loops.
    """
    visited = set()
    cur = model
    while True:
        if hasattr(cur, "layers") and isinstance(cur.layers, (list, torch.nn.ModuleList)):
            return cur
        moved = False
        for attr in ("base_model", "model"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if id(nxt) not in visited and nxt is not cur:
                    visited.add(id(nxt))
                    cur = nxt
                    moved = True
                    break
        if not moved:
            break
    raise ValueError(
        f"Could not find inner Llama model (expected .layers). "
        f"Last node: {type(cur).__name__}, visited {len(visited)} wrappers."
    )
