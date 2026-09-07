"""Build wikitq_tpe_v2_{train,test}.json — WikiTQ with synthetic left_path.

WikiTQ is a flat table with no row-header hierarchy, so the plain
build_wikitq_tpe_dataset.py leaves `left_path` empty for every cell, which makes
TPE's left-path pooling degenerate. This script turns the first column into a
pseudo-left-header instead, so every data cell gets a non-empty
`left_path = [first_col_header, first_col_value]`.

This variant — not the plain one — is what produces the paper's WikiTQ numbers.

v1 (build_wikitq_tpe_dataset.py): top_path=[col_header], left_path=[]
v2 (this script)                : top_path=[col_header], left_path=[first_col_header, first_col_value]

Header row cells: left_path=[] (headers are not part of any data row).
First-column data cells: left_path=[first_col_header, first_col_value] (they ARE
their own row's "left header"; include themselves to avoid the degenerate case
where first-column cells have empty left_path while all others don't).

Inputs  : wikitq_special_{train,test}.json (same as v1)
Outputs : wikitq_tpe_v2_{train,test}.json
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
    """Split a [TAB] [ROW]...[CELL]... table string into rows × cells.

    Mirrors build_wikitq_tpe_dataset.py (v1): cells[row_idx][0] is the row
    prefix (typically empty), cells[row_idx][1..N] are the real column cells.
    So col_id 1..N aligns with tpe_supervised.py's parser.
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
            "left_path": [],  # header row is not part of any data row
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
            # fall back to just the value if header is empty
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
    # Use the same prompts as v1 (copy verbatim from wikitq_special_{name}.json).
    # Only the per_cell_metadata_json changes.
    src = DATA / f"wikitq_special_{name}.json"
    dst = DATA / f"wikitq_tpe_v2_{name}.json"
    print(f"\n=== {name} ===")
    with open(src) as f:
        samples = json.load(f)
    print(f"  input: {src.name}  n={len(samples)}")

    out = []
    n_header_cells = 0
    n_data_cells = 0
    n_nonempty_left = 0
    first_col_header_counter = {}
    first_col_value_counter = {}
    for s in samples:
        prompt = s["prompt"]
        m = TABLE_BLOCK_RE.search(prompt)
        if m is None:
            md = []
        else:
            md = build_metadata(m.group(1))

        for c in md:
            if c["cell_type"] == "header_top":
                n_header_cells += 1
            elif c["cell_type"] == "data":
                n_data_cells += 1
                if c.get("left_path"):
                    n_nonempty_left += 1
        # gather first-col stats from first data cell of first data row
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
    print(f"  distinct first-col headers in this split: {len(first_col_header_counter):,}")
    print(f"  distinct first-col values in this split : {len(first_col_value_counter):,}")
    print(f"  top 8 first-col headers: {sorted(first_col_header_counter.items(), key=lambda x: -x[1])[:8]}")


if __name__ == "__main__":
    process_split("train")
    process_split("test")
