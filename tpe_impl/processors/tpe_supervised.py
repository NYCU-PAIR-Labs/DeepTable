"""TPE-aware supervised dataset preprocessor.

Mirror-copy of `llamafactory.data.processors.supervised.preprocess_supervised_dataset`
with an additional parallel track that emits per-token `top_path_ids` and
`left_path_ids` aligned to the same (row_id, col_id) structure.

Activated only when `data_args.use_tpe` is True. Drop-in replacement for the
existing processor (patched in from `monkey_patch.install_tpe_patch`).

ZERO-INTRUSION: the base `supervised.py` file is untouched. This module is only
imported when `use_tpe=True`.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.data.processors.processor_utils import greedy_knapsack, infer_seqlen
from llamafactory.data.processors.supervised import _encode_supervised_example

from deeptable_paths import DEFAULT_VOCAB_PATH

from .metadata_index import get_metadata_index, prompt_hash

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from llamafactory.hparams import DataArguments
    from llamafactory.data.mm_plugin import ImageInput, VideoInput
    from llamafactory.data.template import Template


logger = logging.get_logger(__name__)

# Path vocabulary — loaded once, lazily
_PATH_VOCAB: Optional[Dict[str, int]] = None
_UNK_ID: int = 1
_PAD_ID: int = 0  # reserved; actual padding uses -1 (collator)


def _load_path_vocab(vocab_path: str) -> Dict[str, int]:
    global _PATH_VOCAB
    if _PATH_VOCAB is None:
        with open(vocab_path) as f:
            meta = json.load(f)
        _PATH_VOCAB = meta["token_to_id"]
        assert meta["pad_id"] == _PAD_ID
        assert meta["unk_id"] == _UNK_ID
        print(f"[tpe] path vocab loaded: {len(_PATH_VOCAB)} entries from {vocab_path}", flush=True)
    return _PATH_VOCAB


def _path_to_ids(path: List[str]) -> List[int]:
    """Map a list of node value strings to vocab ids. Unknown → UNK id."""
    assert _PATH_VOCAB is not None, "call _load_path_vocab() before _path_to_ids()"
    return [_PATH_VOCAB.get(n, _UNK_ID) for n in path]


# -----------------------------------------------------------------------------
# OOV instrumentation
# -----------------------------------------------------------------------------

_OOV_STATS = {
    "total_path_nodes": 0,
    "unk_path_nodes": 0,
    "samples_processed": 0,
    "samples_with_all_oov_cells": 0,
}


def get_oov_stats():
    return dict(_OOV_STATS)


def reset_oov_stats():
    for k in _OOV_STATS:
        _OOV_STATS[k] = 0


# -----------------------------------------------------------------------------
# Main processor
# -----------------------------------------------------------------------------


def preprocess_tpe_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    """TPE-aware version of `preprocess_supervised_dataset`.

    Emits (in addition to the original fields) `top_path_ids` and `left_path_ids`,
    each a per-token list-of-list-of-int of shape [seq_len][variable depth].
    """
    # Sanity: this function should ONLY be called when use_tpe=True
    assert getattr(data_args, "use_tpe", False), \
        "preprocess_tpe_supervised_dataset called with use_tpe=False"
    assert getattr(data_args, "emb_lora", False), \
        "use_tpe requires emb_lora=True (needs [TAB][ROW][CELL] format)"

    # Load vocab (lazy)
    vocab_path = getattr(data_args, "path_vocab_path", None) or DEFAULT_VOCAB_PATH
    _load_path_vocab(vocab_path)

    # Load metadata index (lazy)
    metadata_idx = get_metadata_index()

    model_inputs = {
        "input_ids": [], "attention_mask": [], "labels": [],
        "row_ids": [], "col_ids": [],
        "top_path_ids": [], "left_path_ids": [],
    }

    ori_tokenizer = deepcopy(tokenizer)
    tokenizer.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)

    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue
        if len(examples["_prompt"][i]) != 1:
            raise ValueError("use_tpe expects prompt with one turn")

        prompt_content = examples["_prompt"][i][0]["content"]

        # ------ Out-of-band metadata lookup (fail-fast on miss) ------
        h = prompt_hash(prompt_content)
        if h not in metadata_idx:
            raise RuntimeError(
                f"[tpe] prompt not found in metadata_index. This means the prompt "
                f"differs from hitab_tpe_{{train,test}}.json. Prompt hash prefix: "
                f"{h.hex()[:16]}. First 200 chars: {prompt_content[:200]!r}"
            )
        per_cell_meta = metadata_idx[h]

        # Build quick lookup: (row_id, col_id) -> cell dict
        cell_by_pos: Dict[Tuple[int, int], Dict] = {}
        for c in per_cell_meta:
            cell_by_pos[(c["row_id"], c["col_id"])] = c

        # ------ Same [TAB] extraction as base supervised ------
        pattern = r"/\*\n(.*?)\n\*/"
        def replace_and_extract(text):
            matches = re.findall(pattern, text, re.DOTALL)
            replaced_text = re.sub(pattern, "/*\n[TAB]\n*/", text, flags=re.DOTALL)
            return replaced_text, matches

        examples["_prompt"][i][0]["content"], table_texts = replace_and_extract(prompt_content)

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        if not table_texts or len(table_texts) == 0 or "[ROW]" not in table_texts[0]:
            # No table — emit empty paths for all tokens
            n = len(input_ids)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * n)
            model_inputs["labels"].append(labels)
            model_inputs["row_ids"].append([0] * n)
            model_inputs["col_ids"].append([0] * n)
            model_inputs["top_path_ids"].append([[] for _ in range(n)])
            model_inputs["left_path_ids"].append([[] for _ in range(n)])
            _OOV_STATS["samples_processed"] += 1
            continue

        # ------ Table region: build row/col ids + per-token path ids ------
        # Defaults: non-table tokens get row_id=0, col_id=0, path=[]
        row_ids_total = [0] * len(input_ids)
        col_ids_total = [0] * len(input_ids)
        top_path_total: List[List[int]] = [[] for _ in range(len(input_ids))]
        left_path_total: List[List[int]] = [[] for _ in range(len(input_ids))]

        sample_all_oov_cell_seen = False

        for table_text in table_texts:
            table_token_ids: List[int] = []
            row_ids: List[int] = []
            col_ids: List[int] = []
            top_path: List[List[int]] = []
            left_path: List[List[int]] = []

            parts = table_text.split("[ROW]")
            prefix_text = parts[0]
            if prefix_text:
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                table_token_ids.extend(prefix_ids)
                row_ids.extend([0] * len(prefix_ids))
                col_ids.extend([0] * len(prefix_ids))
                top_path.extend([[] for _ in range(len(prefix_ids))])
                left_path.extend([[] for _ in range(len(prefix_ids))])

            for row_idx, row_text in enumerate(parts[1:]):
                row_id = row_idx
                cell_parts = row_text.split("[CELL]")
                row_pre_ids = tokenizer.encode("[ROW]" + cell_parts[0], add_special_tokens=False)
                table_token_ids.extend(row_pre_ids)
                row_ids.extend([row_id] * len(row_pre_ids))
                col_ids.extend([0] * len(row_pre_ids))
                top_path.extend([[] for _ in range(len(row_pre_ids))])
                left_path.extend([[] for _ in range(len(row_pre_ids))])

                for col_idx, cell_text in enumerate(cell_parts[1:]):
                    col_id = col_idx + 1
                    cell_ids = tokenizer.encode("[CELL]" + cell_text, add_special_tokens=False)
                    # Look up this cell's paths
                    cell = cell_by_pos.get((row_id, col_id))
                    if cell is None:
                        # Pad-rightward cells not in metadata (shouldn't happen if render
                        # is consistent) → empty path, log once
                        cell_top_ids: List[int] = []
                        cell_left_ids: List[int] = []
                    else:
                        cell_top_ids = _path_to_ids(cell["top_path"])
                        cell_left_ids = _path_to_ids(cell["left_path"])
                        # OOV stats
                        n_top = len(cell_top_ids)
                        n_left = len(cell_left_ids)
                        n_unk = (sum(1 for x in cell_top_ids if x == _UNK_ID) +
                                 sum(1 for x in cell_left_ids if x == _UNK_ID))
                        _OOV_STATS["total_path_nodes"] += n_top + n_left
                        _OOV_STATS["unk_path_nodes"] += n_unk
                        if (n_top + n_left) > 0 and n_unk == (n_top + n_left):
                            sample_all_oov_cell_seen = True

                    table_token_ids.extend(cell_ids)
                    row_ids.extend([row_id] * len(cell_ids))
                    col_ids.extend([col_id] * len(cell_ids))
                    top_path.extend([cell_top_ids] * len(cell_ids))
                    left_path.extend([cell_left_ids] * len(cell_ids))

            tab_id = tokenizer.convert_tokens_to_ids("[TAB]")
            if input_ids.count(tab_id) == 0:
                break
            table_insert_idx = input_ids.index(tab_id)
            row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
            col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
            top_path_total = top_path_total[:table_insert_idx] + top_path + top_path_total[table_insert_idx+1:]
            left_path_total = left_path_total[:table_insert_idx] + left_path + left_path_total[table_insert_idx+1:]
            if data_args.train_on_prompt:
                labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
            else:
                labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
            input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]

        # Post-table: if tokenization cutoff truncated, lengths might not match exactly
        # (we may have fewer tokens than row_ids because cutoff_len chopped the prompt)
        n = len(input_ids)
        if len(row_ids_total) != n or len(col_ids_total) != n \
                or len(top_path_total) != n or len(left_path_total) != n:
            raise ValueError(
                f"[tpe] length mismatch: input_ids={n}, row_ids={len(row_ids_total)}, "
                f"col_ids={len(col_ids_total)}, top_path={len(top_path_total)}, "
                f"left_path={len(left_path_total)}"
            )

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * n)
        model_inputs["labels"].append(labels)
        model_inputs["row_ids"].append(row_ids_total)
        model_inputs["col_ids"].append(col_ids_total)
        model_inputs["top_path_ids"].append(top_path_total)
        model_inputs["left_path_ids"].append(left_path_total)
        _OOV_STATS["samples_processed"] += 1
        if sample_all_oov_cell_seen:
            _OOV_STATS["samples_with_all_oov_cells"] += 1

    return model_inputs
