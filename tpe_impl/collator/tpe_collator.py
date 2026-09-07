"""TPE-aware collator — implemented as a monkey-patch of
`DataCollatorForSeq2Seq.__call__`, same pattern as TableLoRA's own collator
patch (`table_lora.py:1585`).

TableLoRA patches `DataCollatorForSeq2Seq.__call__` with `DataCollatorForSeq2Seq_new.__call__`
(which pads row_ids/col_ids). We chain-patch on top: save TableLoRA's patched
version, install our own that (1) calls it, then (2) adds 3D padded
top_path_ids/left_path_ids.

The instance is still a regular `transformers.DataCollatorForSeq2Seq` (a dataclass),
so HF Trainer constructs it normally — only the `__call__` behaviour is altered.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq

PATH_PAD = -1  # sentinel; TPEPathModule.pool() treats -1 as padding

# Module-level OOV counters (read by training callback at epoch end)
_OOV_STATS = {
    "total_path_nodes": 0,
    "unk_path_nodes": 0,
    "batches": 0,
}


def get_oov_stats():
    return dict(_OOV_STATS)


def reset_oov_stats():
    for k in _OOV_STATS:
        _OOV_STATS[k] = 0


# Will be set by install_tpe_collator_patch()
_ORIG_CALL = None
_UNK_ID = 1  # updated from path vocab at install time


def _tpe_call(self, features, return_tensors=None):
    """Patched DataCollatorForSeq2Seq.__call__.

    1. Pop top_path_ids / left_path_ids so TableLoRA's existing padding loop
       (which assumes 1D int lists) doesn't choke on them.
    2. Delegate to the original (TableLoRA-patched) __call__ for everything else.
    3. Reinsert tpe fields as 3D padded tensors.
    """
    # 1. Stash tpe fields (if present)
    popped_top: List[Optional[List[List[int]]]] = []
    popped_left: List[Optional[List[List[int]]]] = []
    any_tpe = False
    for f in features:
        tp = f.pop("top_path_ids", None)
        lp = f.pop("left_path_ids", None)
        popped_top.append(tp)
        popped_left.append(lp)
        if tp is not None:
            any_tpe = True

    # 2. Delegate to original collator (handles input_ids/labels/row_ids/col_ids padding)
    batch = _ORIG_CALL(self, features, return_tensors=return_tensors)

    if not any_tpe:
        return batch

    # 3. 3D pad tpe fields
    B, L = batch["input_ids"].shape
    pad_side = getattr(self.tokenizer, "padding_side", "right")

    # Determine each sample's pre-pad length via attention_mask
    attn = batch["attention_mask"]
    if isinstance(attn, torch.Tensor):
        sample_lens = attn.sum(dim=1).tolist()
    else:
        sample_lens = [int(sum(x)) for x in attn]

    # Token-level alignment: pad each sample's per-token list to L tokens
    top_padded: List[List[List[int]]] = []
    left_padded: List[List[List[int]]] = []
    for b in range(B):
        tp = popped_top[b] or []
        lp = popped_left[b] or []
        orig_len = sample_lens[b]
        # If raw path list is longer than sample_lens (e.g. truncation via
        # tokenizer.pad dropping tail tokens), trim; normally they match exactly.
        tp = list(tp[:orig_len])
        lp = list(lp[:orig_len])
        pad_count = L - len(tp)
        if pad_count > 0:
            pad_block = [[] for _ in range(pad_count)]
            if pad_side == "left":
                tp = pad_block + tp
                lp = pad_block + lp
            else:
                tp = tp + pad_block
                lp = lp + pad_block
        top_padded.append(tp)
        left_padded.append(lp)

    # Depth-level padding with PATH_PAD
    d_top = max(
        (len(ptok) for sample in top_padded for ptok in sample),
        default=0,
    )
    d_left = max(
        (len(ptok) for sample in left_padded for ptok in sample),
        default=0,
    )
    d_top = max(d_top, 1)
    d_left = max(d_left, 1)

    top_arr = np.full((B, L, d_top), PATH_PAD, dtype=np.int64)
    left_arr = np.full((B, L, d_left), PATH_PAD, dtype=np.int64)

    # Populate + OOV stats
    total_nodes = 0
    unk_nodes = 0
    for b in range(B):
        for t in range(L):
            tp = top_padded[b][t]
            lp = left_padded[b][t]
            if tp:
                top_arr[b, t, : len(tp)] = tp
            if lp:
                left_arr[b, t, : len(lp)] = lp
            for x in tp:
                total_nodes += 1
                if x == _UNK_ID:
                    unk_nodes += 1
            for x in lp:
                total_nodes += 1
                if x == _UNK_ID:
                    unk_nodes += 1

    _OOV_STATS["total_path_nodes"] += total_nodes
    _OOV_STATS["unk_path_nodes"] += unk_nodes
    _OOV_STATS["batches"] += 1

    batch["top_path_ids"] = torch.as_tensor(top_arr)
    batch["left_path_ids"] = torch.as_tensor(left_arr)
    return batch


def install_tpe_collator_patch(unk_id: int = 1):
    """Chain-patch DataCollatorForSeq2Seq.__call__ on top of TableLoRA's patch."""
    global _ORIG_CALL, _UNK_ID
    if _ORIG_CALL is not None:
        return  # idempotent
    _ORIG_CALL = DataCollatorForSeq2Seq.__call__
    _UNK_ID = int(unk_id)
    DataCollatorForSeq2Seq.__call__ = _tpe_call
    print(f"[tpe] DataCollatorForSeq2Seq.__call__ monkey-patched (unk_id={unk_id}).", flush=True)


def uninstall_tpe_collator_patch():
    """Revert. Mostly for tests."""
    global _ORIG_CALL
    if _ORIG_CALL is None:
        return
    DataCollatorForSeq2Seq.__call__ = _ORIG_CALL
    _ORIG_CALL = None
    print("[tpe] DataCollatorForSeq2Seq.__call__ restored.", flush=True)
