"""Build combined_path_vocab.json = HiTab vocab ∪ WikiTQ vocab.

HiTab IDs are preserved, so HiTab-only checkpoints stay loadable against the
combined vocab. New WikiTQ-only node values are appended after HiTab's entries.

Schema identical to hitab_tpe_path_vocab.json.

Inputs:
  hitab_tpe_path_vocab.json
  wikitq_tpe_{train,test}.json

Output:
  combined_path_vocab.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR
HITAB_VOCAB = DATA / "hitab_tpe_path_vocab.json"
WIKITQ_TRAIN = DATA / "wikitq_tpe_train.json"
WIKITQ_TEST = DATA / "wikitq_tpe_test.json"
OUT = DATA / "combined_path_vocab.json"


def collect_nodes(path):
    ctr: Counter = Counter()
    with open(path) as f:
        samples = json.load(f)
    for s in samples:
        md = json.loads(s["per_cell_metadata_json"])
        for cell in md:
            for v in cell.get("top_path", []):
                ctr[v] += 1
            for v in cell.get("left_path", []):
                ctr[v] += 1
    return ctr


def main():
    print("Loading HiTab vocab…", flush=True)
    hitab_meta = json.load(HITAB_VOCAB.open())
    hitab_vocab = list(hitab_meta["vocab"])          # preserves order/IDs
    hitab_ids = {v: i for i, v in enumerate(hitab_vocab)}
    print(f"  HiTab vocab size: {len(hitab_vocab)}")

    print("Collecting WikiTQ node values…", flush=True)
    wk_train_ctr = collect_nodes(WIKITQ_TRAIN)
    wk_test_ctr = collect_nodes(WIKITQ_TEST)
    print(f"  WikiTQ train: {sum(wk_train_ctr.values()):,} occurrences / "
          f"{len(wk_train_ctr):,} unique")
    print(f"  WikiTQ test : {sum(wk_test_ctr.values()):,} occurrences / "
          f"{len(wk_test_ctr):,} unique")

    # OOV rate of WikiTQ train vs HiTab-only vocab
    wk_train_oov_occ = sum(c for v, c in wk_train_ctr.items() if v not in hitab_ids)
    wk_train_oov_unique = sum(1 for v in wk_train_ctr if v not in hitab_ids)
    wk_train_total = sum(wk_train_ctr.values())
    print(f"  WikiTQ train OOV vs HiTab-only vocab: "
          f"occ {wk_train_oov_occ:,}/{wk_train_total:,} "
          f"({100*wk_train_oov_occ/max(wk_train_total,1):.2f}%)   "
          f"unique {wk_train_oov_unique:,}/{len(wk_train_ctr):,} "
          f"({100*wk_train_oov_unique/max(len(wk_train_ctr),1):.2f}%)")

    # Build combined vocab: HiTab entries first (IDs preserved), then WikiTQ-only
    new_entries = sorted(
        (v for v in wk_train_ctr if v not in hitab_ids),
        key=lambda v: (-wk_train_ctr[v], v),
    )
    combined = hitab_vocab + new_entries
    combined_ids = {v: i for i, v in enumerate(combined)}
    assert all(combined_ids[v] == hitab_ids[v] for v in hitab_ids), \
        "Combined vocab changed a HiTab entry's ID — refusing to save"

    # OOV of WikiTQ test vs combined vocab (this is what training will actually see)
    wk_test_total = sum(wk_test_ctr.values())
    wk_test_oov_occ = sum(c for v, c in wk_test_ctr.items() if v not in combined_ids)
    wk_test_oov_unique = sum(1 for v in wk_test_ctr if v not in combined_ids)
    print(f"\nCombined vocab size: {len(combined)} "
          f"(HiTab {len(hitab_vocab)} + WikiTQ-only {len(new_entries)})")
    print(f"  WikiTQ test OOV vs combined: "
          f"occ {wk_test_oov_occ:,}/{wk_test_total:,} "
          f"({100*wk_test_oov_occ/max(wk_test_total,1):.2f}%)   "
          f"unique {wk_test_oov_unique:,}/{len(wk_test_ctr):,} "
          f"({100*wk_test_oov_unique/max(len(wk_test_ctr),1):.2f}%)")

    meta = {
        "pad_id": hitab_meta["pad_id"],
        "unk_id": hitab_meta["unk_id"],
        "vocab": combined,
        "token_to_id": combined_ids,
        "hitab_vocab_size": len(hitab_vocab),
        "wikitq_new_entries": len(new_entries),
        "wikitq_test_unk_occurrence_rate": wk_test_oov_occ / max(wk_test_total, 1),
        "wikitq_test_unk_unique_rate": wk_test_oov_unique / max(len(wk_test_ctr), 1),
    }
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:,} bytes)")

    # Sample of the new WikiTQ-only entries
    print("\n=== Sample of new WikiTQ-only vocab entries (by frequency) ===")
    for v in new_entries[:20]:
        print(f"  {wk_train_ctr[v]:>6,} × {v!r}")


if __name__ == "__main__":
    main()
