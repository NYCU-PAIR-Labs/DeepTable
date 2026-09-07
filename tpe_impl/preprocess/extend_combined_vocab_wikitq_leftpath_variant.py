"""Extend combined_path_vocab.json with WikiTQ v2 first-col headers + values.

v2 introduces left_path = [first_col_header, first_col_value] for WikiTQ data
cells. The first_col_header entries mostly overlap with existing top_path
entries (they're column-header strings). The first_col_value entries are new
(cell-level leaves) — similar OOV pattern to HiTab's left_path leaves.

Preserves all existing IDs. Appends new WikiTQ-v2-only entries at the end.
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
    """Return Counter of every node value appearing in top_path OR left_path.

    When also_track_role=True, also returns a dict {role: Counter} where role ∈
    {'top_path', 'left_parent', 'left_leaf'}.
    """
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

    # Collect from v2 train (both top + left)
    train_ctr, train_roles = collect_all_path_nodes(DATA / "wikitq_tpe_v2_train.json", also_track_role=True)
    test_ctr, test_roles = collect_all_path_nodes(DATA / "wikitq_tpe_v2_test.json", also_track_role=True)
    print(f"\nv2 train:")
    print(f"  total path-node occurrences: {sum(train_ctr.values()):,}")
    print(f"  distinct values            : {len(train_ctr):,}")
    print(f"    top_path unique   : {len(train_roles['top_path']):,}")
    print(f"    left_parent unique: {len(train_roles['left_parent']):,}")
    print(f"    left_leaf unique  : {len(train_roles['left_leaf']):,}")

    # OOV of v2 train against EXISTING combined vocab (before extension)
    def oov_stats(ctr, ids):
        total = sum(ctr.values())
        u = sum(1 for v in ctr if v not in ids)
        u_occ = sum(c for v, c in ctr.items() if v not in ids)
        return total, len(ctr), u_occ, u

    for role in ("top_path", "left_parent", "left_leaf"):
        total, unique, u_occ, u_uniq = oov_stats(train_roles[role], existing_ids)
        print(f"  {role} OOV vs current combined vocab: "
              f"{u_occ:,}/{total:,} ({100*u_occ/max(total,1):.2f}%) occ, "
              f"{u_uniq:,}/{unique:,} ({100*u_uniq/max(unique,1):.2f}%) unique")

    # Build extension
    new_entries = sorted(
        ((v, c) for v, c in train_ctr.items() if v not in existing_ids),
        key=lambda x: (-x[1], x[0]),
    )
    new_tokens = [v for v, _ in new_entries]
    final_vocab = existing + new_tokens
    final_ids = {v: i for i, v in enumerate(final_vocab)}
    assert all(final_ids[v] == existing_ids[v] for v in existing_ids), \
        "A pre-existing entry's ID changed — aborting"

    print(f"\nNew entries added by v2: {len(new_tokens):,}")
    print(f"Combined vocab: {len(existing)} → {len(final_vocab)}")

    # OOV of v2 test against FINAL combined vocab
    print("\nv2 test OOV vs FINAL combined vocab:")
    for role in ("top_path", "left_parent", "left_leaf"):
        total, unique, u_occ, u_uniq = oov_stats(test_roles[role], final_ids)
        print(f"  {role:<12} {u_occ:,}/{total:,} ({100*u_occ/max(total,1):.2f}%) occ, "
              f"{u_uniq:,}/{unique:,} ({100*u_uniq/max(unique,1):.2f}%) unique")

    meta["vocab"] = final_vocab
    meta["token_to_id"] = final_ids
    meta["wikitq_v2_new_entries"] = len(new_tokens)
    meta["final_size"] = len(final_vocab)
    VOCAB.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nWrote {VOCAB}  ({VOCAB.stat().st_size:,} bytes)")

    print("\n=== top 15 new WikiTQ-v2 entries ===")
    for v, c in new_entries[:15]:
        print(f"  {c:>7,} × {v!r}")


if __name__ == "__main__":
    main()
