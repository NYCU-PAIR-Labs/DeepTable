"""Render HiTab hmt tree directly to [TAB][ROW][CELL] format + per-cell metadata.

Bypasses TableInstruct's `input_seg` flattening entirely. Produces a canonical
Cartesian-layout render (top_deep-1 header rows + left_deep-1 header cols +
data block), which makes token→(top_path, left_path) alignment 100% deterministic
from hmt structure.

Called by `build_dataset.py`. Pure function, no side effects.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Tree helpers
# -----------------------------------------------------------------------------


def table_deep(node: Dict[str, Any]) -> int:
    """Matches tablelora_preprocess.table_preprocess.hitab_table_type.table_deep.

    Root with no children → depth 1. Root with leaf children → depth 2. Etc.
    """
    ch = node.get("children_dict") or []
    return 1 if not ch else 1 + max(table_deep(c) for c in ch)


def leaves_by_line_idx(root: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """DFS collect every node whose `line_idx` is not None.

    In HiTab any node with `line_idx` corresponds to a rendered line (column in
    top tree, row in left tree). That includes intermediate-with-line_idx nodes
    (subtotal pattern), not only leaves in the graph-theoretic sense.
    """
    out: Dict[int, Dict[str, Any]] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.get("line_idx") is not None:
            out[node["line_idx"]] = node
        for ch in node.get("children_dict") or []:
            stack.append(ch)
    return out


def path_from_root(root: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the ancestor chain from root-level child down to `target`, INCLUDING target.

    Excludes the synthetic root itself (the one with name '<TOP>' / '<LEFT>').
    Uses object identity (id()) to match `target`.

    Returns empty list if target == root or target not found.
    """
    target_id = id(target)

    def walk(node: Dict[str, Any], trail: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if id(node) == target_id:
            return trail + [node]
        for ch in node.get("children_dict") or []:
            found = walk(ch, trail + [node])
            if found is not None:
                return found
        return None

    chain = walk(root, [])
    if chain is None:
        return []
    # chain[0] is root (<TOP>/<LEFT>), strip it
    return chain[1:]


# -----------------------------------------------------------------------------
# Layout constants
# -----------------------------------------------------------------------------

# Render convention:
#   render_row indices (0-indexed):
#     [0, top_deep-2]                → top header rows  (one per tree depth 1..top_deep-1)
#     [top_deep-1, top_deep-1+n_left_leaves-1]  → data rows (one per left-tree line_idx)
#   render_col indices in cell keys (1-indexed; col_id=0 is reserved for [ROW] prefix):
#     [1, left_deep-1]                → left header cols (one per tree depth 1..left_deep-1)
#     [left_deep, left_deep+n_top_leaves-1]  → data cols  (one per top-tree line_idx)
#
# Each cell maps cleanly to either:
#   - header_top     (render_row in top-header band, render_col in data band)
#   - header_left    (render_row in data band,       render_col in left-header band)
#   - data           (render_row in data band,       render_col in data band)
#   - corner         (render_row in top-header band, render_col in left-header band)

CELL_TYPE_HEADER_TOP = "header_top"
CELL_TYPE_HEADER_LEFT = "header_left"
CELL_TYPE_DATA = "data"
CELL_TYPE_CORNER = "corner"


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------


def render_from_hmt(table_data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Render a HiTab hmt JSON to [TAB][ROW][CELL]... string + per-cell metadata.

    Args:
        table_data: parsed hmt JSON with keys top_root, left_root, data, title.

    Returns:
        prompt_string: one-line "[TAB] [ROW] [CELL] ... [ROW] [CELL] ..."
        per_cell_metadata: list of dicts ordered by (row_id, col_id); each dict:
            {
                "row_id": int,                 # 0-indexed, from [ROW] markers
                "col_id": int,                 # 1-indexed, from [CELL] markers within row
                "cell_text": str,
                "top_path": List[str],         # node values from depth-1 descendant down; empty for non-data/non-top-header
                "left_path": List[str],        # same idea for left tree
                "cell_type": "header_top"|"header_left"|"data"|"corner"
            }
    """
    top_root = table_data["top_root"]
    left_root = table_data["left_root"]
    data = table_data.get("data") or []

    top_deep = table_deep(top_root)
    left_deep = table_deep(left_root)
    n_header_rows = top_deep - 1       # top-tree depths 1..top_deep-1
    n_header_cols = left_deep - 1      # left-tree depths 1..left_deep-1

    top_leaves = leaves_by_line_idx(top_root)   # line_idx -> node
    left_leaves = leaves_by_line_idx(left_root)
    n_top_leaves = len(top_leaves)
    n_left_leaves = len(left_leaves)

    # Sanity: every line_idx 0..n-1 should be populated
    for idx in range(n_top_leaves):
        assert idx in top_leaves, f"top tree missing line_idx {idx} (have {sorted(top_leaves.keys())})"
    for idx in range(n_left_leaves):
        assert idx in left_leaves, f"left tree missing line_idx {idx} (have {sorted(left_leaves.keys())})"

    # Pre-compute path_from_root for each leaf (cache to avoid redundant tree traversal)
    top_paths: Dict[int, List[Dict[str, Any]]] = {
        idx: path_from_root(top_root, node) for idx, node in top_leaves.items()
    }
    left_paths: Dict[int, List[Dict[str, Any]]] = {
        idx: path_from_root(left_root, node) for idx, node in left_leaves.items()
    }

    # Sanity: paths must end at the leaf and have length ≤ top_deep - 1 / left_deep - 1
    # (depth of tree = len(path) + 1 since root is depth 0 excluded from path)
    for idx, p in top_paths.items():
        assert len(p) >= 1, f"top leaf {idx} has empty path"
        assert len(p) <= top_deep - 1, f"top leaf {idx} path len {len(p)} exceeds top_deep-1={top_deep-1}"
    for idx, p in left_paths.items():
        assert len(p) >= 1, f"left leaf {idx} has empty path"
        assert len(p) <= left_deep - 1, f"left leaf {idx} path len {len(p)} exceeds left_deep-1={left_deep-1}"

    # ---- build cell grid ----
    total_rows = n_header_rows + n_left_leaves
    total_cols = n_header_cols + n_top_leaves
    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # Top header rows
    for rr in range(n_header_rows):
        depth = rr + 1  # tree depth (1..top_deep-1) this row represents

        # Corner cells (upper-left): empty placeholder with no path
        for ch in range(n_header_cols):
            cc = ch + 1
            cells[(rr, cc)] = {
                "row_id": rr, "col_id": cc,
                "cell_text": "",
                "top_path": [],
                "left_path": [],
                "cell_type": CELL_TYPE_CORNER,
            }

        # Header cells (one per top leaf column)
        for ci in range(n_top_leaves):
            cc = n_header_cols + ci + 1
            path = top_paths[ci]
            # Pick ancestor at this depth; if leaf is shallower, repeat leaf
            idx_in_path = min(depth, len(path)) - 1
            node_here = path[idx_in_path]
            # Path truncated at this depth
            truncated = [str(n["value"]) for n in path[: depth]]
            if len(truncated) < depth and len(truncated) > 0:
                # Shallow-leaf padding: repeat the last node value to match depth
                # (rendering concern; but path metadata stays TRUE length)
                pass  # keep truncated as-is; metadata reflects real ancestry
            cells[(rr, cc)] = {
                "row_id": rr, "col_id": cc,
                "cell_text": str(node_here["value"]),
                "top_path": truncated if truncated else [str(node_here["value"])],
                "left_path": [],
                "cell_type": CELL_TYPE_HEADER_TOP,
            }

    # Data rows (with left-header columns on left)
    for ri in range(n_left_leaves):
        rr = n_header_rows + ri
        left_full = left_paths[ri]

        # Left header cells (one per depth level)
        for ch in range(n_header_cols):
            cc = ch + 1
            depth = ch + 1
            idx_in_path = min(depth, len(left_full)) - 1
            node_here = left_full[idx_in_path]
            truncated = [str(n["value"]) for n in left_full[: depth]]
            cells[(rr, cc)] = {
                "row_id": rr, "col_id": cc,
                "cell_text": str(node_here["value"]),
                "top_path": [],
                "left_path": truncated if truncated else [str(node_here["value"])],
                "cell_type": CELL_TYPE_HEADER_LEFT,
            }

        # Data cells (one per top leaf column)
        for ci in range(n_top_leaves):
            cc = n_header_cols + ci + 1
            top_full = top_paths[ci]
            value = None
            if ri < len(data) and ci < len(data[ri]):
                value = data[ri][ci].get("value")
            # Strip trailing .0 for integer-valued floats (302000.0 → 302000)
            # but leave non-integer floats / strings untouched.
            if value is None:
                text = ""
            elif isinstance(value, float) and value.is_integer():
                text = str(int(value))
            else:
                text = str(value)
            cells[(rr, cc)] = {
                "row_id": rr, "col_id": cc,
                "cell_text": text,
                "top_path": [str(n["value"]) for n in top_full],
                "left_path": [str(n["value"]) for n in left_full],
                "cell_type": CELL_TYPE_DATA,
            }

    # ---- render prompt string ----
    parts = ["[TAB]"]
    for rr in range(total_rows):
        parts.append("[ROW]")
        for cc in range(1, total_cols + 1):
            cell = cells[(rr, cc)]
            # Keep single space between [CELL] and text; empty text → just "[CELL]"
            text = cell["cell_text"]
            if text:
                parts.append(f"[CELL] {text}")
            else:
                parts.append("[CELL]")
    prompt_string = " ".join(parts)

    # ---- flatten metadata in (row_id, col_id) order ----
    metadata = [cells[(rr, cc)] for rr in range(total_rows) for cc in range(1, total_cols + 1)]

    return prompt_string, metadata


# -----------------------------------------------------------------------------
# Convenience: render + wrap in full prompt template
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "This is a hierarchical table question answering task. "
    "The goal for this task is to answer the given question based on the given table. "
    "The table might be hierarchical. "
    "Here is the table to answer this question. Answer the question.\n"
    "/*\n"
    "{table_string}\n"
    "*/\n"
    "Table Caption : {caption}\n"
    "Question : {question}\n"
    "The answer is :\n"
    "\r\n"
)


def build_prompt(table_data: Dict[str, Any], question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Build the full prompt template around the table render.

    Returns (full_prompt, per_cell_metadata).
    """
    table_string, meta = render_from_hmt(table_data)
    caption = table_data.get("title", "")
    if caption:
        caption = f"The table caption is {caption}."
    prompt = PROMPT_TEMPLATE.format(
        table_string=table_string,
        caption=caption,
        question=question,
    )
    return prompt, meta


# -----------------------------------------------------------------------------
# Quick self-check when run as script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from deeptable_paths import HITAB_DIR

    hmt_path = sys.argv[1] if len(sys.argv) > 1 else str(HITAB_DIR / "hmt" / "2744.json")
    with open(hmt_path) as f:
        td = json.load(f)
    ps, md = render_from_hmt(td)
    print("PROMPT STRING:")
    print(ps)
    print()
    print(f"METADATA ({len(md)} cells):")
    for m in md[:25]:
        print(f"  rr={m['row_id']:>2} cc={m['col_id']:>2} type={m['cell_type']:<12} "
              f"text={m['cell_text']!r:<25} top={m['top_path']} left={m['left_path']}")
    print("  ...")
