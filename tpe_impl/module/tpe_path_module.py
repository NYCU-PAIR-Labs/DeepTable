"""TPE-path embedding module (per layer).

For each transformer layer, one `TPEPathModule` contains four `nn.Embedding`s:

  emb_top_path_K   emb_top_path_V
  emb_left_path_K  emb_left_path_V

All zero-init so that when `use_tpe=True` but embeddings untrained,
`pool(path_ids, emb) == 0` → patched forward ≡ original TableLoRA forward
(bit-exact). Verified by tpe_impl/tests/test_bit_exact_equivalence_*.py.

Forward signature:

  top_K, top_V, left_K, left_V = tpe_module(top_path_ids, left_path_ids)

where each output is `[B, L, low_rank_dim]` — ready to sum into the 2D LoRA
pos_signal.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


PATH_PAD = -1  # must match collator.tpe_collator.PATH_PAD


class TPEPathModule(nn.Module):
    """Per-layer hierarchical path embedding module.

    Args:
        path_vocab_size: Total vocab entries INCLUDING PAD and UNK. Pass the
            value from `tpe_config.vocab_size` (i.e. `len(vocab["vocab"])`).
            No `+ 2` arithmetic here — the vocab file already includes PAD/UNK.
        low_rank_dim: Must match TableLoRA's `lora_rank`. Typically 8.
    """

    def __init__(self, path_vocab_size: int, low_rank_dim: int, init_std: float = 0.0,
                 pool_mode: str = "mean", top_only: bool = False):
        super().__init__()
        self.path_vocab_size = int(path_vocab_size)
        self.low_rank_dim = int(low_rank_dim)
        self.init_std = float(init_std)
        if pool_mode not in ("mean", "sum"):
            raise ValueError(f"pool_mode must be 'mean' or 'sum', got {pool_mode!r}")
        self.pool_mode = pool_mode
        self.top_only = bool(top_only)
        self.emb_top_path_K = nn.Embedding(self.path_vocab_size, self.low_rank_dim)
        self.emb_top_path_V = nn.Embedding(self.path_vocab_size, self.low_rank_dim)
        self.emb_left_path_K = nn.Embedding(self.path_vocab_size, self.low_rank_dim)
        self.emb_left_path_V = nn.Embedding(self.path_vocab_size, self.low_rank_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        # Default (init_std=0): zero-init → forward ≡ TableLoRA bit-exact.
        # Non-zero (init_std>0): used by `disable_2d_lora` ablation to break the
        # dead-init chicken-and-egg cycle (without col/row embeds bootstrapping
        # pos_signal off zero, both lora_B_tab and tpe embeddings stay at zero
        # forever — see ablation_no2dlora_report.md).
        for emb in (
            self.emb_top_path_K, self.emb_top_path_V,
            self.emb_left_path_K, self.emb_left_path_V,
        ):
            if self.init_std > 0:
                nn.init.normal_(emb.weight, mean=0.0, std=self.init_std)
            else:
                nn.init.zeros_(emb.weight)

    @staticmethod
    def pool(
        path_ids: torch.Tensor, emb: nn.Embedding,
        precomputed_mask: torch.Tensor = None,
        mode: str = "mean",
    ) -> torch.Tensor:
        """Pool `emb(path_ids)` along the depth dim, skipping PATH_PAD.
        ``mode='mean'`` (default) preserves prior behavior; ``mode='sum'`` skips
        the /denom step so deeper paths get larger magnitude.

        Args:
            path_ids: LongTensor [B, L, D] with values in
                {PATH_PAD (-1), 0..path_vocab_size-1}. -1 marks padding.
            emb: nn.Embedding to look up valid ids.
            precomputed_mask: optional float tensor [B, L, D, 1] (valid=1, pad=0)
                passed in when the SAME mask was already computed this batch
                (k_proj and v_proj share mask per layer, and layers share mask too).

        Returns:
            [B, L, low_rank_dim] float tensor. Tokens with no valid path node
            (all -1) return zero vectors (NOT NaN).
        """
        # Build or reuse mask
        if precomputed_mask is None:
            mask = (path_ids >= 0).to(emb.weight.dtype).unsqueeze(-1)  # [B, L, D, 1]
        else:
            mask = precomputed_mask

        # Replace PAD with 0 so embedding lookup is safe (its contribution is
        # zeroed out by mask)
        safe_ids = path_ids.clamp(min=0)
        vecs = emb(safe_ids)                                          # [B, L, D, dim]
        vecs = vecs * mask                                            # zero-out pads
        if mode == "sum":
            pooled = vecs.sum(dim=-2)                                 # [B, L, dim]
        else:  # mean
            denom = mask.sum(dim=-2).clamp(min=1)                     # [B, L, 1]
            pooled = vecs.sum(dim=-2) / denom                         # [B, L, dim]
        return pooled

    def forward(
        self,
        top_path_ids: torch.Tensor,
        left_path_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (top_K, top_V, left_K, left_V) each of shape [B, L, low_rank_dim]."""
        top_K = self.pool(top_path_ids, self.emb_top_path_K, mode=self.pool_mode)
        top_V = self.pool(top_path_ids, self.emb_top_path_V, mode=self.pool_mode)
        if self.top_only:
            left_K = torch.zeros_like(top_K)
            left_V = torch.zeros_like(top_V)
        else:
            left_K = self.pool(left_path_ids, self.emb_left_path_K, mode=self.pool_mode)
            left_V = self.pool(left_path_ids, self.emb_left_path_V, mode=self.pool_mode)
        return top_K, top_V, left_K, left_V


class TPEPathModuleList(nn.ModuleList):
    """Typed ModuleList of TPEPathModule — one per transformer layer.

    Using a ModuleList ensures every sub-module's parameters are picked up by
    `.parameters()` and state_dict(), and they get a predictable naming scheme
    (`tpe_path_module.{layer_idx}.emb_*.weight`) in state dicts.
    """

    def __init__(self, num_layers: int, path_vocab_size: int, low_rank_dim: int,
                 init_std: float = 0.0, pool_mode: str = "mean", top_only: bool = False):
        super().__init__([
            TPEPathModule(path_vocab_size=path_vocab_size, low_rank_dim=low_rank_dim,
                           init_std=init_std, pool_mode=pool_mode, top_only=top_only)
            for _ in range(num_layers)
        ])
        self.num_layers = int(num_layers)
        self.path_vocab_size = int(path_vocab_size)
        self.low_rank_dim = int(low_rank_dim)
        self.init_std = float(init_std)
        self.pool_mode = pool_mode
        self.top_only = bool(top_only)

    def extra_repr(self) -> str:
        return (f"num_layers={self.num_layers}, "
                f"path_vocab_size={self.path_vocab_size}, "
                f"low_rank_dim={self.low_rank_dim}")
