"""Extend combined_path_vocab.json with FeTaQA v2 first-col headers + values.

Mirrors extend_combined_vocab_wikitq_leftpath_variant.py. FeTaQA v2 introduces
left_path = [first_col_header, first_col_value] for data cells. Most
first_col_header values overlap with existing top_path entries (already in
vocab from FeTaQA v1). first_col_value entries are new leaf strings.

Preserves all existing IDs. Appends new FeTaQA-v2-only entries at the end.
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


def collect_all_path_nodes(path, *, also_track_role=False):
    ctr: Counter = Counter()
    role_ctrs = {"top_path": Counter(), "left_parent": Counter(), "left_leaf": Counter()}
    with open(path) as f:
        samples = json.load(f)
    for s in samples:
        md = json.loads(s["per_cell_metadata_json"])
        for cell in md:
            for v in cell.get("top_path", []):
                ctr[v] += 1
                role_ctrs["top_path"][v] += 1
            lp = cell.get("left_path", [])
            for i, v in enumerate(lp):
                ctr[v] += 1
                if i == 0 and len(lp) >= 2:
                    role_ctrs["left_parent"][v] += 1
                else:
                    role_ctrs["left_leaf"][v] += 1
    if also_track_role:
        return ctr, role_ctrs
    return ctr


def main():
    print("Loading existing combined_path_vocab...", flush=True)
    meta = json.load(VOCAB.open())
    existing = list(meta["vocab"])
    existing_ids = {v: i for i, v in enumerate(existing)}
    print(f"  current size: {len(existing)}")

    train_ctr, train_roles = collect_all_path_nodes(
        DATA / "fetaqa_tpe_v2_train.json", also_track_role=True)
    test_ctr, test_roles = collect_all_path_nodes(
        DATA / "fetaqa_tpe_v2_test.json", also_track_role=True)
    print(f"\nFeTaQA v2 train:")
    print(f"  total path-node occurrences: {sum(train_ctr.values()):,}")
    print(f"  distinct values            : {len(train_ctr):,}")
    print(f"    top_path unique   : {len(train_roles['top_path']):,}")
    print(f"    left_parent unique: {len(train_roles['left_parent']):,}")
    print(f"    left_leaf unique  : {len(train_roles['left_leaf']):,}")

    def oov_stats(ctr, ids):
        total = sum(ctr.values())
        u = sum(1 for v in ctr if v not in ids)
        u_occ = sum(c for v, c in ctr.items() if v not in ids)
        return total, len(ctr), u_occ, u

    for role in ("top_path", "left_parent", "left_leaf"):
        total, unique, u_occ, u_uniq = oov_stats(train_roles[role], existing_ids)
        print(f"  {role:<12} OOV vs current vocab: "
              f"{u_occ:,}/{total:,} ({100*u_occ/max(total,1):.2f}%) occ, "
              f"{u_uniq:,}/{unique:,} ({100*u_uniq/max(unique,1):.2f}%) unique")

    new_entries = sorted(
        ((v, c) for v, c in train_ctr.items() if v not in existing_ids),
        key=lambda x: (-x[1], x[0]),
    )
    new_tokens = [v for v, _ in new_entries]
    final_vocab = existing + new_tokens
    final_ids = {v: i for i, v in enumerate(final_vocab)}
    assert all(final_ids[v] == existing_ids[v] for v in existing_ids), \
        "A pre-existing entry's ID changed — aborting"

    print(f"\nNew entries added by FeTaQA v2: {len(new_tokens):,}")
    print(f"Combined vocab: {len(existing)} → {len(final_vocab)}")

    print("\nFeTaQA v2 test OOV vs FINAL combined vocab:")
    for role in ("top_path", "left_parent", "left_leaf"):
        total, unique, u_occ, u_uniq = oov_stats(test_roles[role], final_ids)
        print(f"  {role:<12} {u_occ:,}/{total:,} ({100*u_occ/max(total,1):.2f}%) occ, "
              f"{u_uniq:,}/{unique:,} ({100*u_uniq/max(unique,1):.2f}%) unique")

    meta["vocab"] = final_vocab
    meta["token_to_id"] = final_ids
    meta["fetaqa_v2_new_entries"] = len(new_tokens)
    meta["final_size"] = len(final_vocab)
    VOCAB.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nWrote {VOCAB}  ({VOCAB.stat().st_size:,} bytes)")

    print("\n=== top 15 new FeTaQA-v2 entries ===")
    for v, c in new_entries[:15]:
        print(f"  {c:>7,} × {v!r}")


if __name__ == "__main__":
    main()
