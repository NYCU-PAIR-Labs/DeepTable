"""Central filesystem-path configuration for the DeepTable pipeline.

Every location the pipeline reads from or writes to is resolved here, so that
nothing in the codebase hard-codes a machine-specific absolute path. Each
location is an environment variable with a repo-relative default that matches
the layout `setup.sh` produces.

Environment variables
---------------------
``DEEPTABLE_LF_ROOT``
    Your LLaMA-Factory checkout. Default: ``<repo>/LLaMA-Factory`` — i.e. the
    clone that ``setup.sh`` creates.
``DEEPTABLE_DATA_DIR``
    Where the preprocessed ``*_special_*.json`` / ``*_tpe_*.json`` files and the
    path vocabularies live. Default: ``$DEEPTABLE_LF_ROOT/data`` (LLaMA-Factory
    resolves ``dataset_info.json`` entries relative to this directory, so the
    two must agree).
``DEEPTABLE_HITAB_DIR``
    Raw HiTab files (``train_samples.jsonl``, ``hitab_test.json``, ``hmt/``).
    Default: ``<repo>/table_preprocess/hitab``.
``DEEPTABLE_RAW_TABLELORA_DIR``
    Only needed by ``build_fetaqa_tabfact_tpe_dataset.py``, which reads the
    bare (non-``_special_``) ``fetaqa_train.json`` / ``tabfact_train.json``
    produced via the compat shim in step 1 of the README.
    Default: ``$DEEPTABLE_DATA_DIR``.
``DEEPTABLE_OUTPUT_DIR``
    Training output / adapter checkpoints. Default: ``$DEEPTABLE_LF_ROOT/output``.

Usage
-----
Scripts inside ``tpe_impl/`` are two levels below the repo root, so they
bootstrap with::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deeptable_paths import DATA_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "LF_ROOT",
    "LF_SRC",
    "DATA_DIR",
    "HITAB_DIR",
    "RAW_TABLELORA_DIR",
    "OUTPUT_DIR",
    "DEFAULT_VOCAB_PATH",
    "tpe_dataset_files",
    "require_file",
    "require_dir",
]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


#: Root of this repository (the directory containing this file).
REPO_ROOT = Path(__file__).resolve().parent

#: LLaMA-Factory checkout that the overlay in `llamafactory_src_overlay/` is applied to.
LF_ROOT = _env_path("DEEPTABLE_LF_ROOT", REPO_ROOT / "LLaMA-Factory")

#: Importable source root of that checkout (`sys.path` entry for `import llamafactory`).
LF_SRC = LF_ROOT / "src"

#: Preprocessed datasets + path vocabularies. Must match LLaMA-Factory's `data/`.
DATA_DIR = _env_path("DEEPTABLE_DATA_DIR", LF_ROOT / "data")

#: Raw HiTab files: train_samples.jsonl, hitab_test.json, hmt/{table_id}.json
HITAB_DIR = _env_path("DEEPTABLE_HITAB_DIR", REPO_ROOT / "table_preprocess" / "hitab")

#: Bare (non-`_special_`) TableLoRA-format FeTaQA/TabFact files, see README step 1.
RAW_TABLELORA_DIR = _env_path("DEEPTABLE_RAW_TABLELORA_DIR", DATA_DIR)

#: Training outputs / adapter checkpoints.
OUTPUT_DIR = _env_path("DEEPTABLE_OUTPUT_DIR", LF_ROOT / "output")

#: Default path-node vocabulary. Single-benchmark (HiTab-only) runs use this;
#: multi-benchmark runs should point `path_vocab_path` at combined_path_vocab.json.
DEFAULT_VOCAB_PATH = str(DATA_DIR / "hitab_tpe_path_vocab.json")


def tpe_dataset_files() -> list:
    """The per-cell-metadata files that `processors/metadata_index.py` indexes.

    HiTab uses the plain metadata; WikiTQ / FeTaQA / TabFact use the
    `*_tpe_v2_*` files built by the `*_leftpath_variant.py` scripts. The v1
    (`wikitq_tpe_*`, `fetaqa_tpe_*`, `tabfact_tpe_*`) files are deliberately
    NOT listed: they carry the same prompts as their v2 counterparts but
    different metadata, so indexing both would be a prompt-hash conflict.
    Set `DEEPTABLE_DATA_DIR` to relocate all of them at once.
    """
    names = [
        "hitab_tpe_train.json",
        "hitab_tpe_test.json",
        "wikitq_tpe_v2_train.json",
        "wikitq_tpe_v2_test.json",
        "fetaqa_tpe_v2_train.json",
        "fetaqa_tpe_v2_test.json",
        "tabfact_tpe_v2_train.json",
        "tabfact_tpe_v2_test.json",
    ]
    return [str(DATA_DIR / n) for n in names]


def _missing(kind: str, path, hint: str) -> str:
    return (
        f"{kind} not found: {path}\n"
        f"  {hint}\n"
        f"  Current path settings (override with environment variables):\n"
        f"    DEEPTABLE_LF_ROOT   = {LF_ROOT}\n"
        f"    DEEPTABLE_DATA_DIR  = {DATA_DIR}\n"
        f"    DEEPTABLE_HITAB_DIR = {HITAB_DIR}"
    )


def require_file(path, hint: str = "See README.md 'How to reproduce'."):
    """Return `path` if it exists, else raise FileNotFoundError naming the env vars."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(_missing("File", p, hint))
    return p


def require_dir(path, hint: str = "See README.md 'How to reproduce'."):
    """Return `path` if it is an existing directory, else raise FileNotFoundError."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(_missing("Directory", p, hint))
    return p
