"""Stratify test-set path-node OOV rate by depth in path.

Depth convention (index into path list):
  depth 0 = path[0] = the root's direct child (group label, e.g. "percent", "games")
  depth 1 = path[1] = child of that
  depth N = path[N] = the leaf / lowest node for that cell

Per-node position gets its OOV% computed separately for top_path and left_path.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR
TRAIN = DATA / "hitab_tpe_train.json"
TEST = DATA / "hitab_tpe_test.json"
VOCAB = DATA / "hitab_tpe_path_vocab.json"


def main():
    vocab = set(json.load(VOCAB.open())["vocab"])
    test = json.load(TEST.open())

    # Counters: (tree, depth_index_in_path) -> (total, unknown)
    totals: dict = defaultdict(int)
    unks: dict = defaultdict(int)

    # Also: per-cell OOV (how many cells have ALL path OOV vs some OOV)
    cell_all_oov = 0
    cell_any_oov = 0
    cell_total = 0

    for sample in test:
        md = json.loads(sample["per_cell_metadata_json"])
        for cell in md:
            if cell["cell_type"] not in ("data", "header_top", "header_left"):
                continue
            for tree_name, path in [("top", cell["top_path"]), ("left", cell["left_path"])]:
                for depth_idx, node in enumerate(path):
                    totals[(tree_name, depth_idx)] += 1
                    if node not in vocab:
                        unks[(tree_name, depth_idx)] += 1

            # Per-cell stats on data cells only
            if cell["cell_type"] == "data":
                full_path = cell["top_path"] + cell["left_path"]
                if full_path:
                    cell_total += 1
                    n_unk = sum(1 for n in full_path if n not in vocab)
                    if n_unk == len(full_path):
                        cell_all_oov += 1
                    if n_unk > 0:
                        cell_any_oov += 1

    print("=== OOV rate by path depth (test set) ===\n")
    print(f"{'tree':<6} {'depth':>6} {'total':>10} {'unk':>10} {'unk %':>8}")
    for key in sorted(totals.keys()):
        tree, d = key
        t = totals[key]
        u = unks[key]
        pct = 100 * u / t if t else 0
        print(f"{tree:<6} {d:>6} {t:>10,} {u:>10,} {pct:>7.2f}%")

    print(f"\n=== Per-data-cell OOV (test set, {cell_total:,} data cells) ===")
    print(f"  any-OOV   cells: {cell_any_oov:>8,} ({100*cell_any_oov/max(cell_total,1):.2f}%)  "
          f"— at least one path node is OOV")
    print(f"  all-OOV   cells: {cell_all_oov:>8,} ({100*cell_all_oov/max(cell_total,1):.2f}%)  "
          f"— ALL path nodes OOV (cell effectively falls back to row/col only)")


if __name__ == "__main__":
    main()
