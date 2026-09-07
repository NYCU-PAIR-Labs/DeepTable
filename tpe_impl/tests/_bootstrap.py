"""Shared setup for the scripts in `tpe_impl/tests/`.

These are run directly, e.g.::

    python tpe_impl/tests/test_tpe_path_module.py

This module puts the repo root and the LLaMA-Factory source root on `sys.path`
(see `deeptable_paths` for how those are configured), and provides `require_*`
helpers so that a test which needs data, a GPU, or a trained adapter that the
user does not have prints a SKIP line and exits 0 instead of crashing.

Only `test_tpe_path_module.py` runs with no external prerequisites at all; the
others need at least the preprocessed HiTab files from README step 2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deeptable_paths import DATA_DIR, DEFAULT_VOCAB_PATH, LF_SRC, OUTPUT_DIR  # noqa: E402

if LF_SRC.is_dir() and str(LF_SRC) not in sys.path:
    sys.path.insert(0, str(LF_SRC))

#: Backbone used by the tests that load a real model. Override with
#: DEEPTABLE_BACKBONE to test against a different one (e.g. a Qwen2 checkpoint).
BACKBONE = os.environ.get("DEEPTABLE_BACKBONE", "deepseek-ai/deepseek-llm-7b-chat")

__all__ = [
    "BACKBONE",
    "DATA_DIR",
    "DEFAULT_VOCAB_PATH",
    "OUTPUT_DIR",
    "skip",
    "require_data",
    "require_vocab",
    "require_gpu",
    "require_adapter",
]


def skip(reason: str):
    """Print a SKIP line and exit 0 — a missing prerequisite is not a failure."""
    print(f"SKIP: {reason}")
    raise SystemExit(0)


def require_data(name: str) -> Path:
    """Return `$DEEPTABLE_DATA_DIR/name`, or skip if it hasn't been built yet."""
    p = Path(DATA_DIR) / name
    if not p.is_file():
        skip(
            f"{name} not found in {DATA_DIR}. Build it with the "
            f"tpe_impl/preprocess/ scripts (README step 2), or set "
            f"DEEPTABLE_DATA_DIR to the directory holding it."
        )
    return p


def require_vocab(name: str = None) -> str:
    """Return the path-node vocabulary path, or skip if it hasn't been built."""
    if name is None:
        p = Path(DEFAULT_VOCAB_PATH)
        if not p.is_file():
            skip(
                f"{p} not found. Build it with "
                f"tpe_impl/preprocess/build_hitab_path_vocab.py (README step 2)."
            )
        return str(p)
    return str(require_data(name))


def require_gpu():
    """Skip unless a CUDA device is available."""
    import torch

    if not torch.cuda.is_available():
        skip("no CUDA device available; this test loads a model onto the GPU.")


def require_adapter(env_var: str = "DEEPTABLE_TEST_ADAPTER") -> str:
    """Return a trained adapter directory, or skip.

    Reads `$DEEPTABLE_TEST_ADAPTER`; falls back to nothing, since no adapter
    ships with this repo — the caller must point at one they trained
    themselves (README step 4).
    """
    d = os.environ.get(env_var)
    if not d or not Path(d).is_dir():
        skip(
            f"no trained adapter available. Set {env_var} to an adapter "
            f"directory produced by README step 4 (e.g. "
            f"{Path(OUTPUT_DIR) / 'hitab_sab'})."
        )
    return d
