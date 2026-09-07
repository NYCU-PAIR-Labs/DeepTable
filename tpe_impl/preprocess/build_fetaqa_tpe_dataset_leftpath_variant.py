"""Build fetaqa_tpe_v2_{train,test}.json — FeTaQA with synthetic left_path.

Same idea as build_wikitq_tpe_dataset_leftpath_variant.py — FeTaQA is a flat
table, so turn the first column into a pseudo-left-header and give every data
cell a non-empty `left_path = [first_col_header, first_col_value]`.

This variant — not the plain one — is what produces the paper's FeTaQA numbers.

v1 (build_fetaqa_tabfact_tpe_dataset.py): top_path=[col_header], left_path=[]
v2 (this)                               : top_path=[col_header], left_path=[first_col_header, first_col_value]

Reads the v1 file (already in `[TAB][ROW][CELL]` canonical format) and ONLY
rewrites per_cell_metadata_json. The prompt is preserved verbatim, so v1 and v2
hash identically in metadata_index — which is why only v2 is listed in
deeptable_paths.tpe_dataset_files().

Header row cells: left_path=[]
First-column data cells: left_path=[first_col_header, first_col_value]
  (include themselves to avoid degenerate first-col=empty case)

Outputs: fetaqa_tpe_v2_{train,test}.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR
TABLE_BLOCK_RE = re.compile(r"/\*\n(.*?)\n\*/", re.DOTALL)


def parse_table(table_text: str):
    """Split `[TAB] [ROW] ... [CELL] ...` into rows × cells.

    Mirrors build_wikitq_tpe_dataset_leftpath_variant.py —
    build_fetaqa_tabfact_tpe_dataset.py inserts an empty prefix cell at index 0
    of each row, so cells[r][0] is the prefix, cells[r][1..N] are real cells.
    col_id = col_idx aligns with
    tpe_supervised.py's parser (which uses col_id starting at 1).
    """
    parts = table_text.split("[ROW]")
    rows = []
    for row_text in parts[1:]:
        cells = row_text.split("[CELL]")
        rows.append([c.strip() for c in cells])
    return rows


def build_metadata(table_text: str):
    rows = parse_table(table_text)
    if not rows:
        return []
    header_cells = rows[0]  # [prefix, h1, h2, ...]
    first_col_header = header_cells[1] if len(header_cells) > 1 else ""

    n_cols = max(len(r) - 1 for r in rows)
    col_headers = [""] * (n_cols + 1)
    for c in range(1, len(header_cells)):
        col_headers[c] = header_cells[c]

    metadata = []
    # Header row
    for col_idx in range(1, len(header_cells)):
        text = header_cells[col_idx]
        metadata.append({
            "row_id": 0,
            "col_id": col_idx,
            "cell_text": text,
            "top_path": [text] if text else [],
            "left_path": [],
            "cell_type": "header_top",
        })
    # Data rows
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        first_col_val = row[1].strip() if len(row) > 1 else ""
        left_path = []
        if first_col_header and first_col_val:
            left_path = [first_col_header, first_col_val]
        elif first_col_val:
            left_path = [first_col_val]
        for col_idx in range(1, len(row)):
            text = row[col_idx]
            col_hdr = col_headers[col_idx] if col_idx < len(col_headers) else ""
            metadata.append({
                "row_id": row_idx,
                "col_id": col_idx,
                "cell_text": text,
                "top_path": [col_hdr] if col_hdr else [],
                "left_path": list(left_path),
                "cell_type": "data",
            })
    return metadata


def process_split(name: str):
    src = DATA / f"fetaqa_tpe_{name}.json"
    dst = DATA / f"fetaqa_tpe_v2_{name}.json"
    print(f"\n=== fetaqa {name} ===")
    with open(src) as f:
        samples = json.load(f)
    print(f"  input: {src.name}  n={len(samples)}")

    out = []
    n_header_cells = 0
    n_data_cells = 0
    n_nonempty_left = 0
    n_no_table = 0
    first_col_header_counter = {}
    first_col_value_counter = {}
    for s in samples:
        prompt = s["prompt"]
        m = TABLE_BLOCK_RE.search(prompt)
        if m is None:
            md = []
            n_no_table += 1
        else:
            md = build_metadata(m.group(1))

        for c in md:
            if c["cell_type"] == "header_top":
                n_header_cells += 1
            elif c["cell_type"] == "data":
                n_data_cells += 1
                if c.get("left_path"):
                    n_nonempty_left += 1
        # gather first-col stats
        data_cells = [c for c in md if c.get("cell_type") == "data" and c.get("left_path")]
        if data_cells:
            lp = data_cells[0]["left_path"]
            if len(lp) >= 1:
                first_col_header_counter[lp[0]] = first_col_header_counter.get(lp[0], 0) + 1
            if len(lp) >= 2:
                first_col_value_counter[lp[1]] = first_col_value_counter.get(lp[1], 0) + 1

        out.append({
            "prompt": prompt,
            "response": s["response"],
            "per_cell_metadata_json": json.dumps(md, ensure_ascii=False),
        })

    with open(dst, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  output: {dst.name}  size={dst.stat().st_size:,}")
    print(f"  header_top cells: {n_header_cells:,}")
    print(f"  data cells:       {n_data_cells:,}  "
          f"(of which {n_nonempty_left:,} have non-empty left_path = "
          f"{100*n_nonempty_left/max(n_data_cells,1):.1f}%)")
    print(f"  samples without /* table */ block: {n_no_table}")
    print(f"  distinct first-col headers: {len(first_col_header_counter):,}")
    print(f"  distinct first-col values:  {len(first_col_value_counter):,}")
    print(f"  top 8 first-col headers: {sorted(first_col_header_counter.items(), key=lambda x: -x[1])[:8]}")


if __name__ == "__main__":
    process_split("train")
    process_split("test")
