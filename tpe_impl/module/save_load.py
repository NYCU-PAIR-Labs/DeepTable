"""Save / load the TPEPathModuleList state. PeftModel's adapter save only
handles `lora_*` prefixed params; our `tpe_path_module.*` params won't make
it into `adapter_model.safetensors`. So we save them separately (pattern
borrowed from `llamafactory.sab.sab.save_sab / load_sab_state`).

Files written into `<checkpoint_dir>`:
  tpe_modules.safetensors         — flat state_dict of the ModuleList
  tpe_path_vocab.json             — COPY of the path vocab used during training
                                     (so load-time can verify size consistency)
  tpe_metadata.json               — {num_layers, path_vocab_size, low_rank_dim,
                                     vocab_digest}  used for size assertion on load

On load:
  1. Read tpe_metadata.json → know expected shape
  2. Read tpe_path_vocab.json → get vocab entries
  3. Assert len(vocab.vocab) == expected vocab_size  [prevents silent corruption]
  4. Instantiate TPEPathModuleList with those dims
  5. Load tpe_modules.safetensors weights
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file as _st_load
from safetensors.torch import save_file as _st_save


TPE_MODULES_FILENAME = "tpe_modules.safetensors"
TPE_VOCAB_FILENAME = "tpe_path_vocab.json"
TPE_META_FILENAME = "tpe_metadata.json"

# Pre-rename filenames (from this module's "Hier-LoRA" / "hier" codename era —
# see README "Naming"). save_tpe_state() never writes these; load_tpe_state()
# falls back to them so checkpoints trained before the rename still load.
_LEGACY_FILENAMES = {
    TPE_MODULES_FILENAME: "hier_modules.safetensors",
    TPE_VOCAB_FILENAME: "hier_path_vocab.json",
    TPE_META_FILENAME: "hier_metadata.json",
}


def _resolve_checkpoint_file(load_dir: Path, current_name: str) -> Path:
    """Return `load_dir/current_name`, falling back to its pre-rename filename."""
    current = load_dir / current_name
    if current.exists():
        return current
    legacy = load_dir / _LEGACY_FILENAMES[current_name]
    return legacy if legacy.exists() else current


def save_tpe_state(
    model,
    save_dir: str,
    tpe_config,
) -> str:
    """Write tpe_path_module's state_dict + vocab copy + metadata.

    Returns the modules file path.
    """
    if not hasattr(model, "tpe_path_module"):
        print("[tpe] save_tpe_state: model has no tpe_path_module, skipping.")
        return ""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Flatten state dict (prefix "tpe_path_module." already present in model's
    # state_dict but we save ONLY the module's own state to avoid bloat)
    module_list = model.tpe_path_module
    state = {k: v.detach().cpu().clone().contiguous() for k, v in module_list.state_dict().items()}
    modules_path = save_dir / TPE_MODULES_FILENAME
    _st_save(state, str(modules_path))

    # Copy vocab file
    with open(tpe_config.path_vocab_path) as f:
        vocab = json.load(f)
    (save_dir / TPE_VOCAB_FILENAME).write_text(json.dumps(vocab, ensure_ascii=False))

    # Metadata
    meta = {
        "num_layers": module_list.num_layers,
        "path_vocab_size": module_list.path_vocab_size,
        "low_rank_dim": module_list.low_rank_dim,
        "vocab_digest": tpe_config.vocab_file_digest,
    }
    (save_dir / TPE_META_FILENAME).write_text(json.dumps(meta, indent=2))

    n_tensors = len(state)
    print(f"[tpe] save_tpe_state wrote {n_tensors} tensors to {modules_path} "
          f"(vocab_size={meta['path_vocab_size']}, dim={meta['low_rank_dim']}, "
          f"num_layers={meta['num_layers']})", flush=True)
    return str(modules_path)


def load_tpe_state(
    model,
    load_dir: str,
    strict: bool = True,
) -> bool:
    """Load tpe_path_module state from `<load_dir>`.

    Returns True on success, False if files absent (silent skip).

    The model must ALREADY have `tpe_path_module` attached (by install_tpe_patch
    with matching dims). This function only loads the weights.

    Raises AssertionError on size mismatch (prevents silent corruption).
    """
    load_dir = Path(load_dir)
    modules_path = _resolve_checkpoint_file(load_dir, TPE_MODULES_FILENAME)
    meta_path = _resolve_checkpoint_file(load_dir, TPE_META_FILENAME)
    vocab_path = _resolve_checkpoint_file(load_dir, TPE_VOCAB_FILENAME)

    if not modules_path.exists():
        if strict:
            raise FileNotFoundError(
                f"Neither {TPE_MODULES_FILENAME} nor its pre-rename name "
                f"({_LEGACY_FILENAMES[TPE_MODULES_FILENAME]}) exists in {load_dir} "
                f"— cannot load tpe state. Did you train with use_tpe=True?"
            )
        return False
    if modules_path.name != TPE_MODULES_FILENAME:
        print(f"[tpe] load_tpe_state: {load_dir} uses pre-rename filenames "
              f"({modules_path.name}); loading those.", flush=True)

    # Metadata sanity
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        actual_vocab_size = len(vocab["vocab"])
        if "path_vocab_size" in meta:
            assert actual_vocab_size == meta["path_vocab_size"], (
                f"tpe_path_vocab.json length ({actual_vocab_size}) "
                f"!= tpe_metadata.json path_vocab_size ({meta['path_vocab_size']}). "
                f"Checkpoint is inconsistent — refuse to load."
            )

    # Verify ModuleList dims match
    if hasattr(model, "tpe_path_module"):
        module_list = model.tpe_path_module
        if "num_layers" in meta:
            assert module_list.num_layers == meta["num_layers"], (
                f"num_layers mismatch: model={module_list.num_layers} "
                f"vs checkpoint={meta['num_layers']}"
            )
        if "path_vocab_size" in meta:
            assert module_list.path_vocab_size == meta["path_vocab_size"], (
                f"path_vocab_size mismatch: model={module_list.path_vocab_size} "
                f"vs checkpoint={meta['path_vocab_size']}"
            )
        if "low_rank_dim" in meta:
            assert module_list.low_rank_dim == meta["low_rank_dim"], (
                f"low_rank_dim mismatch: model={module_list.low_rank_dim} "
                f"vs checkpoint={meta['low_rank_dim']}"
            )
    else:
        raise RuntimeError(
            "model.tpe_path_module not attached. Call install_tpe_patch() first."
        )

    # Load tensors
    state = _st_load(str(modules_path))
    # Cast dtype/device to match model's ModuleList params
    sample_param = next(module_list.parameters())
    target_dtype = sample_param.dtype
    target_device = sample_param.device
    state = {k: v.to(device=target_device, dtype=target_dtype) for k, v in state.items()}

    result = module_list.load_state_dict(state, strict=strict)
    # result is an IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    if strict and (result.missing_keys or result.unexpected_keys):
        raise AssertionError(
            f"load_tpe_state strict load failed: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )

    # Quick stats after load
    with torch.no_grad():
        absmax = torch.stack([
            m.emb_top_path_K.weight.abs().max()
            for m in module_list
        ]).max().item()
    print(f"[tpe] load_tpe_state: loaded {len(state)} tensors from {modules_path}; "
          f"emb_top_path_K absmax={absmax:.4e}", flush=True)
    return True
