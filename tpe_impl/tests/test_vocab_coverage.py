"""Sanity check: vocab coverage + unknown rate.

A light assertion that the path-node vocabulary built by
`preprocess/build_hitab_path_vocab.py` is consistent with the training data it
was built from, plus a re-print of the coverage / unknown-rate numbers.
`preprocess/analyze_oov_depth.py` gives the same numbers broken down by depth.
"""
import json, sys
from collections import Counter
from pathlib import Path

from _bootstrap import DATA_DIR, require_data, require_vocab  # noqa: F401

DATA = Path(DATA_DIR)


def collect(dataset):
    ctr = Counter()
    for s in dataset:
        md = json.loads(s["per_cell_metadata_json"])
        for c in md:
            for v in c.get("top_path", []):
                ctr[v] += 1
            for v in c.get("left_path", []):
                ctr[v] += 1
    return ctr


def main():
    TRAIN = require_data("hitab_tpe_train.json")
    TEST = require_data("hitab_tpe_test.json")
    VOCAB = Path(require_vocab())

    vocab_meta = json.load(VOCAB.open())
    vocab = set(vocab_meta["vocab"])
    print(f"Vocab size (incl PAD + UNK): {len(vocab)}")
    print(f"  pad_id: {vocab_meta['pad_id']}  unk_id: {vocab_meta['unk_id']}")

    train = json.load(TRAIN.open())
    test = json.load(TEST.open())
    print(f"Train samples: {len(train)}  Test samples: {len(test)}")

    train_ctr = collect(train)
    test_ctr = collect(test)

    # Train coverage should be 100% by construction
    train_unk = sum(c for v, c in train_ctr.items() if v not in vocab)
    train_total = sum(train_ctr.values())
    print(f"\nTrain OOV (should be 0): {train_unk}/{train_total} "
          f"({100*train_unk/max(train_total,1):.4f}%)")
    assert train_unk == 0, f"Train should have 0 OOV, got {train_unk}"

    test_unk = sum(c for v, c in test_ctr.items() if v not in vocab)
    test_total = sum(test_ctr.values())
    test_unk_unique = sum(1 for v in test_ctr if v not in vocab)
    print(f"Test OOV by occurrence: {test_unk}/{test_total} "
          f"({100*test_unk/max(test_total,1):.2f}%)")
    print(f"Test OOV by unique:     {test_unk_unique}/{len(test_ctr)} "
          f"({100*test_unk_unique/max(len(test_ctr),1):.2f}%)")

    # Per-data-cell all-OOV rate
    cell_all_oov = 0
    cell_any_oov = 0
    cell_total_data = 0
    for s in test:
        md = json.loads(s["per_cell_metadata_json"])
        for c in md:
            if c["cell_type"] != "data":
                continue
            path = c["top_path"] + c["left_path"]
            if not path:
                continue
            cell_total_data += 1
            n_unk = sum(1 for v in path if v not in vocab)
            if n_unk > 0:
                cell_any_oov += 1
            if n_unk == len(path):
                cell_all_oov += 1
    print(f"\nPer-data-cell (test, n={cell_total_data:,}):")
    print(f"  any-OOV: {cell_any_oov:,} ({100*cell_any_oov/max(cell_total_data,1):.2f}%)")
    print(f"  all-OOV: {cell_all_oov:,} ({100*cell_all_oov/max(cell_total_data,1):.2f}%)")

    # Acceptance: test_unk_occurrence < 25%, all-OOV < 5%
    test_unk_rate = test_unk / max(test_total, 1)
    all_oov_rate = cell_all_oov / max(cell_total_data, 1)
    assert test_unk_rate < 0.25, f"Test OOV occurrence {test_unk_rate:.2%} > 25%"
    assert all_oov_rate < 0.05, f"All-OOV data cells {all_oov_rate:.2%} > 5%"

    print("\n✓ Sanity 2 vocab coverage passed.")


if __name__ == "__main__":
    main()
