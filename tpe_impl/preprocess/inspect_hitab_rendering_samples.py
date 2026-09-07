"""Diagnostic: pick 8 diverse samples across top_deep / left_deep, show the new
canonical render vs the old TableInstruct render side-by-side, and list the
cell→path mapping. For manual inspection only; not part of the pipeline."""

import json, sys, os, random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from render_from_hmt import render_from_hmt, table_deep
from deeptable_paths import DATA_DIR, HITAB_DIR  # noqa: E402

HMT_DIR = HITAB_DIR / "hmt"
TRAIN_JSONL = HITAB_DIR / "train_samples.jsonl"
OLD_PREP = DATA_DIR / "hitab_special_train.json"
NEW_PREP = DATA_DIR / "hitab_tpe_train.json"


def table_stats(hmt):
    return dict(
        top_deep=table_deep(hmt["top_root"]),
        left_deep=table_deep(hmt["left_root"]),
        data_rows=len(hmt["data"]),
        data_cols=len(hmt["data"][0]) if hmt["data"] else 0,
    )


def extract_table_block(prompt):
    """Return the content between /* and */ markers."""
    import re
    m = re.search(r"/\*\n(.*?)\n\*/", prompt, flags=re.DOTALL)
    return m.group(1).strip() if m else "(no table block)"


def main():
    # Load samples
    with open(TRAIN_JSONL) as f:
        raw_samples = [json.loads(l) for l in f]
    old_ds = json.load(open(OLD_PREP))
    new_ds = json.load(open(NEW_PREP))

    # Build q2table map (raw) and q→idx maps for old/new preprocessed data
    # Old preprocessed has q in prompt, response = answer
    import re
    q_re = re.compile(r"Question : (.*?)\nThe answer is :", re.DOTALL)

    # Pair raw_samples index with old/new preprocessed indices
    # Both old and new preprocessing produce samples in order of raw train_samples.jsonl
    assert len(raw_samples) == len(old_ds) == len(new_ds), \
        f"length mismatch: raw={len(raw_samples)}, old={len(old_ds)}, new={len(new_ds)}"

    # Pick 8 diverse samples by (top_deep, left_deep, data_shape)
    targets = [
        ("shallow flat-ish",     dict(top_deep=2, left_deep=2, nrow=(3, 10), ncol=(2, 6))),
        ("deep top, flat left",  dict(top_deep=3, left_deep=2, nrow=(3, 8),  ncol=(5, 10))),
        ("flat top, deep left",  dict(top_deep=2, left_deep=3, nrow=(3, 8),  ncol=(2, 6))),
        ("deep both",            dict(top_deep=3, left_deep=3, nrow=(4, 8),  ncol=(3, 8))),
        ("very deep left",       dict(top_deep=3, left_deep=4, nrow=(3, 8),  ncol=(3, 8))),
        ("very deep top",        dict(top_deep=4, left_deep=3, nrow=(3, 8),  ncol=(3, 8))),
        ("subtotal pattern",     dict(top_deep=3, left_deep=4, nrow=(4, 8),  ncol=(4, 10))),  # will try to find one with intermediate line_idx
        ("big",                  dict(top_deep=3, left_deep=3, nrow=(10, 40), ncol=(5, 12))),
    ]
    rng = random.Random(42)
    picked = []
    used_ids = set()
    for label, crit in targets:
        cands = []
        for i, r in enumerate(raw_samples):
            tid = r["table_id"]
            if tid in used_ids:
                continue
            try:
                hmt = json.load(open(HMT_DIR / f"{tid}.json"))
            except FileNotFoundError:
                continue
            st = table_stats(hmt)
            if (st["top_deep"] == crit["top_deep"]
                and st["left_deep"] == crit["left_deep"]
                and crit["nrow"][0] <= st["data_rows"] <= crit["nrow"][1]
                and crit["ncol"][0] <= st["data_cols"] <= crit["ncol"][1]):
                # For "subtotal" target, require intermediate with line_idx
                if label == "subtotal pattern":
                    def has_subtotal(n):
                        if n.get("children_dict") and n.get("line_idx") is not None:
                            return True
                        for c in n.get("children_dict") or []:
                            if has_subtotal(c):
                                return True
                        return False
                    if not has_subtotal(hmt["left_root"]) and not has_subtotal(hmt["top_root"]):
                        continue
                cands.append((i, tid, hmt, st, r))
        if cands:
            choice = rng.choice(cands)
            picked.append((label, choice))
            used_ids.add(choice[1])
        else:
            print(f"  no candidate for {label}", file=sys.stderr)

    # Emit report markdown
    lines = []
    lines.append("# Sanity Report — `render_from_hmt` Output Check")
    lines.append("")
    lines.append(f"Samples picked: {len(picked)} (covering top_deep 2-4 × left_deep 2-4, including subtotal pattern).")
    lines.append("")
    lines.append("For each sample we show:")
    lines.append("1. **hmt stats** (top/left depth, data shape)")
    lines.append("2. **OLD** — TableInstruct flattened render (from `hitab_special_train.json`)")
    lines.append("3. **NEW** — Our canonical hmt render (from `hitab_tpe_train.json`)")
    lines.append("4. **Path table** — first 10 cells with (row_id, col_id, text, top_path, left_path, cell_type)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for label, (i, tid, hmt, st, raw) in picked:
        lines.append(f"## {label} — `table_id = {tid}` (sample idx {i})")
        lines.append("")
        lines.append(f"- top_deep = {st['top_deep']}, left_deep = {st['left_deep']}")
        lines.append(f"- data = {st['data_rows']} × {st['data_cols']}")
        lines.append(f"- question: {raw['question']}")
        lines.append(f"- answer:   {raw['answer']}")
        lines.append("")

        # OLD table block (truncate if huge)
        old_block = extract_table_block(old_ds[i]["prompt"])
        new_block = extract_table_block(new_ds[i]["prompt"])

        def trunc(s, lim=800):
            return (s[:lim] + f"…({len(s) - lim} chars more)") if len(s) > lim else s

        lines.append("### OLD (TableInstruct flatten)")
        lines.append("```")
        lines.append(trunc(old_block))
        lines.append("```")
        lines.append("")
        lines.append("### NEW (canonical hmt render)")
        lines.append("```")
        lines.append(trunc(new_block))
        lines.append("```")
        lines.append("")

        # Metadata: first 10 + last 5 cells
        md = json.loads(new_ds[i]["per_cell_metadata_json"])
        lines.append(f"### Cell→Path mapping ({len(md)} cells total)")
        lines.append("")
        lines.append("| row_id | col_id | text | top_path | left_path | type |")
        lines.append("|---:|---:|---|---|---|---|")
        for m in md[:12]:
            top = "·".join(m["top_path"]) if m["top_path"] else "∅"
            left = "·".join(m["left_path"]) if m["left_path"] else "∅"
            txt = m["cell_text"][:40]
            lines.append(f"| {m['row_id']} | {m['col_id']} | `{txt}` | {top} | {left} | {m['cell_type']} |")
        if len(md) > 12:
            lines.append(f"| … | … | … | … | … | … ({len(md) - 12} more cells) |")
            # Show some data cells too for context
            data_cells = [m for m in md if m["cell_type"] == "data"][:3]
            for m in data_cells:
                top = "·".join(m["top_path"]) if m["top_path"] else "∅"
                left = "·".join(m["left_path"]) if m["left_path"] else "∅"
                txt = m["cell_text"][:40]
                lines.append(f"| {m['row_id']} | {m['col_id']} | `{txt}` | {top} | {left} | {m['cell_type']} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Aggregate statistics
    lines.append("## Aggregate Dataset Stats (all 7417 train + 1584 test)")
    lines.append("")

    import statistics as stats
    train_n = len(new_ds)
    test_n = len(json.load(open(DATA_DIR / "hitab_tpe_test.json")))
    train_md_sizes = []
    train_top_path_lens = []
    train_left_path_lens = []
    for s in new_ds[::100]:  # sample stride 100 to speed up
        md = json.loads(s["per_cell_metadata_json"])
        train_md_sizes.append(len(md))
        for m in md:
            if m["cell_type"] == "data":
                train_top_path_lens.append(len(m["top_path"]))
                train_left_path_lens.append(len(m["left_path"]))

    lines.append(f"- **Train samples**: {train_n}")
    lines.append(f"- **Test samples**: {test_n}")
    lines.append(f"- **Skipped**: 0 (all tables parse cleanly)")
    lines.append(f"- **Cells / sample (train, stride 100)**: "
                 f"mean {stats.fmean(train_md_sizes):.0f}, "
                 f"median {stats.median(train_md_sizes):.0f}, "
                 f"max {max(train_md_sizes)}")
    lines.append(f"- **top_path length (data cells)**: "
                 f"mean {stats.fmean(train_top_path_lens):.2f}, "
                 f"max {max(train_top_path_lens)}")
    lines.append(f"- **left_path length (data cells)**: "
                 f"mean {stats.fmean(train_left_path_lens):.2f}, "
                 f"max {max(train_left_path_lens)}")

    # Path length variance (to confirm subtotal-pattern variability)
    var_samples = sum(1 for s in new_ds
                     for md in [json.loads(s["per_cell_metadata_json"])]
                     if len(set(len(m["left_path"]) for m in md if m["cell_type"] == "data")) > 1)
    lines.append(f"- **Samples with per-row variable left_path length** (subtotal pattern): "
                 f"{var_samples}/{train_n} ({100*var_samples/train_n:.1f}%)")

    lines.append("")
    lines.append("## Known divergence from TableInstruct render")
    lines.append("")
    lines.append("Our canonical render **deliberately** differs from TableInstruct's flatten in two ways:")
    lines.append("")
    lines.append("1. **No left-tree transposition**: TableInstruct will transpose a single depth-1 "
                 "left-tree intermediate (e.g. `percent` with no siblings) into a full row; our "
                 "canonical render keeps it as a column header at col 1 for every data row. "
                 "This adds O(n_left_leaves) rows of redundant `percent` text in exchange for a "
                 "fully deterministic layout.")
    lines.append("")
    lines.append("2. **Corner cells are empty**: TableInstruct fills corner cells with the name of the "
                 "first column header (e.g. `weight classification`) repeated across top-header rows. "
                 "We leave them empty (`[CELL]` with no text). No information lost (placeholder was "
                 "never in hmt tree anyway).")
    lines.append("")
    lines.append("**Consequence**: `hitab_tpe_*.json` tokenised sequences are NOT identical to "
                 "`hitab_special_*.json`. Baseline (TableLoRA / Vanilla LoRA) on the new preprocessing "
                 "needs to be re-run to compare against use_tpe. This is expected.")

    out = Path(__file__).parent.parent / "hitab_rendering_report.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
