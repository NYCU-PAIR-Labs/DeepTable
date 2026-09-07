"""Extend combined_path_vocab.json to include FeTaQA + TabFact column headers.

Preserves existing HiTab + WikiTQ entries at their current IDs. New FeTaQA /
TabFact column header values are appended in order (frequency desc).

Input  : combined_path_vocab.json
Output : same path (overwrite).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR
VOCAB = DATA / "combined_path_vocab.json"


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
    print("Loading current combined vocab...", flush=True)
    meta = json.load(VOCAB.open())
    existing_vocab = list(meta["vocab"])
    existing_ids = {v: i for i, v in enumerate(existing_vocab)}
    print(f"  current size: {len(existing_vocab)} (HiTab {meta.get('hitab_vocab_size')}"
          f" + WikiTQ-only {meta.get('wikitq_new_entries')})")

    oov_stats = {}
    new_counters = {}
    for ds in ("fetaqa", "tabfact"):
        train = collect_nodes(DATA / f"{ds}_tpe_train.json")
        test = collect_nodes(DATA / f"{ds}_tpe_test.json")
        total = sum(train.values()); unique = len(train)
        unk_occ = sum(c for v, c in train.items() if v not in existing_ids)
        unk_uniq = sum(1 for v in train if v not in existing_ids)
        test_total = sum(test.values())
        test_unk_occ = sum(c for v, c in test.items() if v not in existing_ids and v not in train)
        print(f"\n  {ds} train: {total:,} occ / {unique:,} unique")
        print(f"    OOV vs current combined: {unk_occ:,}/{total:,} ({100*unk_occ/max(total,1):.2f}%) occurrences, "
              f"{unk_uniq:,}/{unique:,} ({100*unk_uniq/max(unique,1):.2f}%) unique")
        oov_stats[ds] = {
            "train_total": total, "train_unique": unique,
            "train_unk_occ": unk_occ, "train_unk_uniq": unk_uniq,
            "test_total": test_total,
        }
        new_counters[ds] = train

    # Merge all new entries (prefer entries we've already got, dedupe across datasets)
    combined_new = Counter()
    for ds, ctr in new_counters.items():
        for v, c in ctr.items():
            if v not in existing_ids:
                combined_new[v] += c
    new_entries = sorted(combined_new.items(), key=lambda x: (-x[1], x[0]))
    new_tokens = [v for v, _ in new_entries]
    print(f"\n  new (FeTaQA + TabFact) unique entries added: {len(new_tokens)}")

    final_vocab = existing_vocab + new_tokens
    final_ids = {v: i for i, v in enumerate(final_vocab)}
    assert all(final_ids[v] == existing_ids[v] for v in existing_ids), \
        "A pre-existing entry's ID changed — aborting"

    # Compute per-dataset OOV stats against FINAL combined vocab
    final_oov = {}
    for ds in ("fetaqa", "tabfact"):
        test = collect_nodes(DATA / f"{ds}_tpe_test.json")
        test_total = sum(test.values())
        test_unk_occ = sum(c for v, c in test.items() if v not in final_ids)
        test_unk_uniq = sum(1 for v in test if v not in final_ids)
        final_oov[ds] = {
            "test_total_occ": test_total,
            "test_unk_occ": test_unk_occ,
            "test_unk_occ_rate": test_unk_occ / max(test_total, 1),
            "test_unk_uniq": test_unk_uniq,
            "test_unique": len(test),
        }
        print(f"\n  {ds} test OOV vs FINAL combined vocab: "
              f"{test_unk_occ:,}/{test_total:,} ({100*test_unk_occ/max(test_total,1):.2f}%) occurrences, "
              f"{test_unk_uniq:,}/{len(test):,} ({100*test_unk_uniq/max(len(test),1):.2f}%) unique")

    meta["vocab"] = final_vocab
    meta["token_to_id"] = final_ids
    meta["fetaqa_tabfact_new_entries"] = len(new_tokens)
    meta["final_size"] = len(final_vocab)
    meta["extended_oov_stats"] = final_oov
    VOCAB.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nWrote {VOCAB}  final size = {len(final_vocab)}  ({VOCAB.stat().st_size:,} bytes)")

    print("\n=== Sample of new entries ===")
    for v, c in new_entries[:15]:
        print(f"  {c:>7,} × {v!r}")


if __name__ == "__main__":
    main()
