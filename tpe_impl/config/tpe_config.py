"""TPE configuration reader.

Reads use_tpe-related fields from a training yaml without touching
`llamafactory/hparams/data_args.py`. Defaults are no-op (use_tpe=False),
so if the yaml doesn't mention use_tpe the module returns an inert config
and install_tpe_patch() becomes a no-op.

Usage:
    from tpe_impl.config import TPEConfig
    cfg = TPEConfig.from_yaml("configs/hitab_tpe.yaml")
    if cfg.use_tpe:
        install_tpe_patch(model, cfg)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DEFAULT_VOCAB_PATH  # noqa: E402


@dataclass
class TPEConfig:
    """All use_tpe-related knobs.

    Loaded from yaml keys:
      use_tpe:           bool      # master flag; default False
      path_vocab_path:     str       # path to the path-node vocabulary json
      path_low_rank_dim:   int       # must match TableLoRA's lora_rank (8)
      path_vocab_size:     int|None  # cached size; if given, assert-checked against vocab file
      tpe_lr_multiplier:  float     # optional separate lr for tpe params (like SAB lr100x)
      disable_2d_lora:     bool      # ablation: drop col_embeds + row_embeds from pos_signal
      pool_mode:           str       # 'mean' (default) or 'sum' for tpe path pooling
      tpe_top_only:       bool      # ablation: zero-out left_pool, keep top_pool only
    """

    use_tpe: bool = False
    path_vocab_path: str = DEFAULT_VOCAB_PATH
    path_low_rank_dim: int = 8
    path_vocab_size: Optional[int] = None  # None = read from vocab file
    tpe_lr_multiplier: float = 1.0
    disable_2d_lora: bool = False
    pool_mode: str = "mean"
    tpe_top_only: bool = False

    # Derived / loaded at runtime
    _vocab: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TPEConfig":
        """Read use_tpe-related keys from a training yaml. Keys not present → defaults."""
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
        kwargs = {}
        if "use_tpe" in cfg:
            kwargs["use_tpe"] = bool(cfg["use_tpe"])
        if "path_vocab_path" in cfg:
            kwargs["path_vocab_path"] = str(cfg["path_vocab_path"])
        if "path_low_rank_dim" in cfg:
            kwargs["path_low_rank_dim"] = int(cfg["path_low_rank_dim"])
        if "path_vocab_size" in cfg:
            kwargs["path_vocab_size"] = int(cfg["path_vocab_size"])
        if "tpe_lr_multiplier" in cfg:
            kwargs["tpe_lr_multiplier"] = float(cfg["tpe_lr_multiplier"])
        if "disable_2d_lora" in cfg:
            kwargs["disable_2d_lora"] = bool(cfg["disable_2d_lora"])
        if "pool_mode" in cfg:
            kwargs["pool_mode"] = str(cfg["pool_mode"])
        if "tpe_top_only" in cfg:
            kwargs["tpe_top_only"] = bool(cfg["tpe_top_only"])
        return cls(**kwargs)

    @classmethod
    def from_env(cls) -> "TPEConfig":
        """Env-var overrides (mainly for quick testing).

        Preferred names are USE_TPE / TPE_PATH_VOCAB / TPE_DIM / TPE_LR_MULT.
        The TPE_LORA* spellings are accepted as aliases for backwards
        compatibility with older run scripts.
        """
        env = os.environ

        def _first(*names):
            for n in names:
                if n in env:
                    return env[n]
            return None

        kwargs = {}
        raw_use = _first("USE_TPE", "TPE_LORA")
        if raw_use is not None:
            kwargs["use_tpe"] = raw_use.lower() in ("1", "true", "yes")
        raw_vocab = _first("TPE_PATH_VOCAB", "TPE_LORA_PATH_VOCAB")
        if raw_vocab is not None:
            kwargs["path_vocab_path"] = raw_vocab
        raw_dim = _first("TPE_DIM", "TPE_LORA_DIM")
        if raw_dim is not None:
            kwargs["path_low_rank_dim"] = int(raw_dim)
        raw_mult = _first("TPE_LR_MULT")
        if raw_mult is not None:
            kwargs["tpe_lr_multiplier"] = float(raw_mult)
        return cls(**kwargs)

    def load_vocab(self) -> dict:
        """Lazy-load the path vocab JSON. Returns the full meta dict."""
        if self._vocab is None:
            with open(self.path_vocab_path) as f:
                self._vocab = json.load(f)
            # Size sanity: if user supplied path_vocab_size, check it matches
            if self.path_vocab_size is not None:
                actual = len(self._vocab["vocab"])
                assert actual == self.path_vocab_size, (
                    f"path_vocab_size mismatch: config says {self.path_vocab_size}, "
                    f"vocab file has {actual}"
                )
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Total entries including PAD and UNK."""
        return len(self.load_vocab()["vocab"])

    @property
    def vocab_file_digest(self) -> str:
        """Short hash for checkpoint identity check (to prevent silent corruption)."""
        import hashlib
        with open(self.path_vocab_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience: find the yaml the user launched training with
# ---------------------------------------------------------------------------

def tpe_config_from_training_args(training_args_or_yaml) -> TPEConfig:
    """Utility for integration with LLaMA-Factory's training loop.

    If `training_args_or_yaml` is a string ending in .yaml/.yml, parse as yaml.
    Otherwise try `training_args.config` (if present) or fall back to env.
    """
    if isinstance(training_args_or_yaml, str) and \
       (training_args_or_yaml.endswith(".yaml") or training_args_or_yaml.endswith(".yml")):
        return TPEConfig.from_yaml(training_args_or_yaml)
    cfg_path = getattr(training_args_or_yaml, "config", None) or \
               getattr(training_args_or_yaml, "yaml_path", None)
    if cfg_path and os.path.exists(cfg_path):
        return TPEConfig.from_yaml(cfg_path)
    return TPEConfig.from_env()
