"""Build wikitq_tpe_train.json / wikitq_tpe_test.json for the TPE pipeline.

WikiTQ is a flat-header table dataset (single header row, no row-header hierarchy).
We emit the same per-cell metadata schema as hitab_tpe_*.json so the existing
TPE pipeline (collator, supervised processor, module) works unchanged.

Degenerate mapping:
  - row 0 cells                 → cell_type="header_top", top_path=[cell_text], left_path=[]
  - row ≥1 cells                → cell_type="data",        top_path=[header_at_(0,col)], left_path=[]

The prompt string itself is copied verbatim from wikitq_special_{train,test}.json,
so the prompt hash lookup in processors/tpe_supervised.py works identically.

Inputs:
  wikitq_special_{train,test}.json

Outputs:
  wikitq_tpe_{train,test}.json  (same prompts +
    per_cell_metadata_json)
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

    Returns list[list[str]]: rows[row_idx][col_idx] = cell_text (col 0 is the
    row prefix text which is typically empty/whitespace, matching the
    tpe_supervised col_id = col_idx+1 convention).
    """
    # Strip leading [TAB] wordy prefix
    parts = table_text.split("[ROW]")
    # parts[0] = anything before first [ROW] (contains "[TAB]" and maybe spaces)
    # parts[1:] = row bodies
    rows = []
    for row_text in parts[1:]:
        cells = row_text.split("[CELL]")
        # cells[0] = row prefix (pre first [CELL]), cells[1:] = actual cells
        # Mirror tpe_supervised.py: col_id 1..N correspond to cells[1..N]
        rows.append([c.strip() for c in cells])
    return rows


def build_metadata(table_text: str):
    """Produce per-cell metadata for a WikiTQ flat table.

    Convention matches HiTab: row_id/col_id correspond to the (row_idx, col_idx+1)
    positions produced by tpe_supervised.py's parser.
    """
    rows = parse_table(table_text)
    if not rows:
        return []

    # Row 0 is the header row. Col index 0 is the row prefix (empty),
    # col index 1..N are the column cells.
    header_cells = rows[0]
    # header_cells[0] is the row prefix (usually ""); columns are header_cells[1:]
    n_cols = max(len(r) - 1 for r in rows)  # max columns across rows
    col_headers = [""] * (n_cols + 1)
    for col_idx in range(1, len(header_cells)):
        col_headers[col_idx] = header_cells[col_idx]

    metadata = []
    # Emit header row cells
    for col_idx in range(1, len(header_cells)):
        cell_text = header_cells[col_idx]
        metadata.append({
            "row_id": 0,
            "col_id": col_idx,
            "cell_text": cell_text,
            "top_path": [cell_text] if cell_text else [],
            "left_path": [],
            "cell_type": "header_top",
        })
    # Emit data row cells
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        for col_idx in range(1, len(row)):
            cell_text = row[col_idx]
            col_hdr = col_headers[col_idx] if col_idx < len(col_headers) else ""
            metadata.append({
                "row_id": row_idx,
                "col_id": col_idx,
                "cell_text": cell_text,
                "top_path": [col_hdr] if col_hdr else [],
                "left_path": [],
                "cell_type": "data",
            })
    return metadata


def process_split(name: str):
    in_path = DATA / f"wikitq_special_{name}.json"
    out_path = DATA / f"wikitq_tpe_{name}.json"
    print(f"\n=== processing {name} ===")
    with open(in_path) as f:
        samples = json.load(f)
    print(f"  input: {in_path.name}  n={len(samples)}")

    out = []
    n_skipped_no_table = 0
    n_header_cells = 0
    n_data_cells = 0
    sample_cell_counts = []
    for s in samples:
        prompt = s["prompt"]
        m = TABLE_BLOCK_RE.search(prompt)
        if m is None:
            # No /* ... */ table block — emit empty metadata; tpe_supervised
            # will fall through to the "no table" branch.
            n_skipped_no_table += 1
            md = []
        else:
            table_text = m.group(1)
            md = build_metadata(table_text)
        n_header_cells += sum(1 for c in md if c["cell_type"] == "header_top")
        n_data_cells += sum(1 for c in md if c["cell_type"] == "data")
        sample_cell_counts.append(len(md))
        out.append({
            "prompt": prompt,
            "response": s["response"],
            "per_cell_metadata_json": json.dumps(md, ensure_ascii=False),
        })

    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  output: {out_path.name}  size={out_path.stat().st_size:,}")
    print(f"  header_top cells: {n_header_cells:,}")
    print(f"  data cells:       {n_data_cells:,}")
    print(f"  samples without /* table */ block: {n_skipped_no_table}")
    if sample_cell_counts:
        import statistics
        print(f"  per-sample cell count — "
              f"mean {statistics.mean(sample_cell_counts):.1f}  "
              f"median {statistics.median(sample_cell_counts):.0f}  "
              f"min {min(sample_cell_counts)}  "
              f"max {max(sample_cell_counts)}")
    return out


if __name__ == "__main__":
    process_split("train")
    process_split("test")
