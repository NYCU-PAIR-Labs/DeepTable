"""Build fetaqa_tpe_{train,test}.json and tabfact_tpe_{train,test}.json.

Raw data (the bare, non-`_special_` fetaqa/tabfact json produced via the compat
shim in README step 1; located via $DEEPTABLE_RAW_TABLELORA_DIR) uses the
`<COL>/<ROW>` separator format. This script:

  1. Parses the /* ... */ table block from each sample's prompt
  2. Re-serialises it to our `[TAB] [ROW] [CELL] ...` canonical format
     (matches HiTab / WikiTQ tpe preprocessing)
  3. Generates per_cell_metadata with:
       - row 0 cells → cell_type="header_top", top_path=[cell_text]
       - row ≥1 cells → cell_type="data",        top_path=[col_header_at_(0,col)]
       - left_path = [] for all (flat tables, no left-header hierarchy)
  4. Writes the new prompt + response + per_cell_metadata_json

Outputs ($DEEPTABLE_DATA_DIR):
  fetaqa_tpe_{train,test}.json
  tabfact_tpe_{train,test}.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR, RAW_TABLELORA_DIR  # noqa: E402

RAW_DIR = RAW_TABLELORA_DIR
OUT_DIR = DATA_DIR
TABLE_BLOCK_RE = re.compile(r"/\*\n(.*?)\n\*/", re.DOTALL)


def parse_cols_rows(table_text: str):
    """Raw table uses `<COL>` as cell separator, `<ROW>` as row separator.

    Returns list[list[str]]: cells[row_idx][col_idx].
    Unlike WikiTQ's [TAB][ROW][CELL] where col 0 is a row prefix, here cells[0..N]
    are all real cells. We normalise by inserting an empty row-prefix cell so
    the downstream Cartesian mapping (col_id = col_idx+1) matches HiTab/WikiTQ.
    """
    rows = table_text.split("<ROW>")
    out = []
    for r in rows:
        cells = r.split("<COL>")
        # Prepend empty prefix so col_id 1..N map to cells[1..N]
        out.append([""] + [c.strip() for c in cells])
    return out


def rebuild_prompt(prompt: str, new_table_block: str) -> str:
    # Use a lambda so backslashes in `new_table_block` are treated literally
    # (re.sub otherwise interprets \1, \g<...> etc. as regex group refs).
    return TABLE_BLOCK_RE.sub(lambda _m: f"/*\n{new_table_block}\n*/", prompt, count=1)


def rows_to_tab_format(rows) -> str:
    """Turn rows (with prefix empties) back into `[TAB] [ROW] [CELL] v1 [CELL] v2 ...`"""
    parts = ["[TAB]"]
    for r in rows:
        parts.append("[ROW]")
        # skip prefix cell (rows[i][0] = "")
        for c in r[1:]:
            parts.append("[CELL]")
            parts.append(c)
    return " ".join(parts)


def build_metadata(rows):
    """Per-cell metadata. rows[i][j] = cell text, j>=1 are real cells.

    Flat table convention:
      - row 0 (header) cells: cell_type=header_top, top_path=[cell_text], left_path=[]
      - row >=1 data cells:   cell_type=data, top_path=[col_header_at_(0,j)], left_path=[]
    """
    if not rows:
        return []
    header = rows[0]  # [prefix, h1, h2, ...]
    metadata = []
    # header row
    for col_idx in range(1, len(header)):
        h = header[col_idx]
        metadata.append({
            "row_id": 0,
            "col_id": col_idx,
            "cell_text": h,
            "top_path": [h] if h else [],
            "left_path": [],
            "cell_type": "header_top",
        })
    # data rows
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        for col_idx in range(1, len(row)):
            text = row[col_idx]
            col_hdr = header[col_idx] if col_idx < len(header) else ""
            metadata.append({
                "row_id": row_idx,
                "col_id": col_idx,
                "cell_text": text,
                "top_path": [col_hdr] if col_hdr else [],
                "left_path": [],
                "cell_type": "data",
            })
    return metadata


def process_split(dataset_name: str, split: str):
    """dataset_name ∈ {'fetaqa', 'tabfact'}. split ∈ {'train', 'test'}."""
    src = RAW_DIR / f"{dataset_name}_{split}.json"
    dst = OUT_DIR / f"{dataset_name}_tpe_{split}.json"
    print(f"\n=== {dataset_name} {split} ===")
    with open(src) as f:
        samples = json.load(f)
    print(f"  input: {src.name}  n={len(samples)}")

    out = []
    n_header_cells = 0
    n_data_cells = 0
    n_no_table = 0
    row_counts = []
    col_counts = []
    for s in samples:
        prompt = s["prompt"]
        m = TABLE_BLOCK_RE.search(prompt)
        if m is None:
            n_no_table += 1
            md = []
            new_prompt = prompt
        else:
            raw_table = m.group(1)
            rows = parse_cols_rows(raw_table)
            tab_format_block = rows_to_tab_format(rows)
            new_prompt = rebuild_prompt(prompt, tab_format_block)
            md = build_metadata(rows)
            row_counts.append(len(rows))
            col_counts.append(max((len(r) for r in rows), default=0) - 1)  # subtract prefix

        n_header_cells += sum(1 for c in md if c["cell_type"] == "header_top")
        n_data_cells += sum(1 for c in md if c["cell_type"] == "data")
        out.append({
            "prompt": new_prompt,
            "response": s["response"],
            "per_cell_metadata_json": json.dumps(md, ensure_ascii=False),
        })

    with open(dst, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  output: {dst.name}  size={dst.stat().st_size:,}")
    print(f"  header_top cells: {n_header_cells:,}")
    print(f"  data cells:       {n_data_cells:,}")
    print(f"  samples without /* table */ block: {n_no_table}")
    if row_counts:
        import statistics
        print(f"  rows per sample — mean {statistics.mean(row_counts):.1f}  median {statistics.median(row_counts):.0f}  "
              f"min {min(row_counts)}  max {max(row_counts)}")
        print(f"  cols per sample — mean {statistics.mean(col_counts):.1f}  median {statistics.median(col_counts):.0f}  "
              f"min {min(col_counts)}  max {max(col_counts)}")


if __name__ == "__main__":
    for ds in ("fetaqa", "tabfact"):
        for split in ("train", "test"):
            process_split(ds, split)
