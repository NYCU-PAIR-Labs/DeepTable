"""Out-of-band metadata lookup index.

LLaMA-Factory's aligner (`convert_alpaca`) drops extra columns like
`per_cell_metadata_json`, so we can't rely on it reaching the processor
through `examples[...]`. Instead we load the hitab_tpe_*.json files
independently once and build a prompt-hash → metadata dict.

This keeps zero-intrusion on the aligner.

Usage:
    idx = get_metadata_index()  # lazy-builds on first call
    meta = idx[sha256_hash_of_prompt]        # returns per_cell_metadata list
                                              # (fail-fast on miss in tpe_supervised)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR, tpe_dataset_files  # noqa: E402

# The per-cell-metadata files to index. Relocate them all at once by setting
# DEEPTABLE_DATA_DIR; see deeptable_paths.tpe_dataset_files() for why the
# v1 WikiTQ/FeTaQA/TabFact files are excluded (same prompts as their v2
# counterparts but different metadata, so indexing both would be ambiguous).
_DEFAULT_FILES = tpe_dataset_files()

_INDEX: Optional[Dict[bytes, List[Dict]]] = None
_INDEX_LOCK = threading.Lock()


def prompt_hash(prompt: str) -> bytes:
    """SHA-256(prompt) truncated to 16 bytes (128 bits).

    Collision probability for ~10k samples: negligible (~10k²/2^128 = 3e-31).
    Deterministic across processes (unlike Python's hash()).
    """
    return hashlib.sha256(prompt.encode("utf-8")).digest()[:16]


def _build_index(paths) -> Dict[bytes, List[Dict]]:
    idx: Dict[bytes, List[Dict]] = {}
    found = []
    for p in paths:
        if not os.path.exists(p):
            continue
        found.append(p)
        with open(p) as f:
            rows = json.load(f)
        for r in rows:
            h = prompt_hash(r["prompt"])
            md = json.loads(r["per_cell_metadata_json"])
            # If the same prompt appears in multiple datasets, ensure metadata matches
            if h in idx and idx[h] != md:
                raise RuntimeError(
                    f"metadata_index: prompt hash collision with DIFFERENT metadata "
                    f"across {paths}. This should not happen with SHA-256 on distinct prompts."
                )
            idx[h] = md

    if not found:
        raise FileNotFoundError(
            "metadata_index: none of the per-cell-metadata files exist, so TPE has "
            "no metadata to attach to any prompt.\n"
            f"  Looked in: {DATA_DIR}\n"
            f"  Expected at least one of: {[os.path.basename(p) for p in paths]}\n"
            "  Build them with the tpe_impl/preprocess/build_*_tpe_dataset*.py scripts "
            "(README step 2), or point DEEPTABLE_DATA_DIR at the directory holding them."
        )
    return idx


def get_metadata_index(extra_paths=None) -> Dict[bytes, List[Dict]]:
    """Lazy-init global metadata index.

    Scans `deeptable_paths.tpe_dataset_files()` by default; pass `extra_paths`
    to index additional `*_tpe_*.json` files on top of those. Raises
    FileNotFoundError if not a single one of them exists.
    """
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                paths = list(_DEFAULT_FILES)
                if extra_paths:
                    for p in extra_paths:
                        if p not in paths:
                            paths.append(p)
                _INDEX = _build_index(paths)
                loaded = [p for p in paths if os.path.exists(p)]
                print(f"[tpe] metadata_index built: {len(_INDEX)} unique prompts "
                      f"from {len(loaded)} file(s): "
                      f"{[os.path.basename(p) for p in loaded]}", flush=True)
    return _INDEX


def reset_for_tests():
    """Clear the global index. Only for unit tests."""
    global _INDEX
    _INDEX = None
