"""Smoke test for the TPE data collator: run it on a batch of 2 and verify the
[B, L, D_max] shape with -1 padding."""
import json, sys, os
from pathlib import Path

from _bootstrap import BACKBONE, require_data, require_gpu, require_vocab  # noqa: F401

import torch
from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

from tpe_impl.processors.tpe_supervised import preprocess_tpe_supervised_dataset
from tpe_impl.processors.metadata_index import reset_for_tests
from tpe_impl.collator import (
    install_tpe_collator_patch, uninstall_tpe_collator_patch,
    get_oov_stats, PATH_PAD,
)
# Ensure TableLoRA's collator patch is applied first (it normally runs when
# load_table_lora() is called in real training). For test, apply it manually.
from llamafactory.table_lora.table_lora import (
    DataCollatorForSeq2Seq_new, load_table_lora,
)
from transformers import DataCollatorForSeq2Seq

MODEL = BACKBONE  # override with $DEEPTABLE_BACKBONE


def main():
    reset_for_tests()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left",
                                              add_eos_token=True, add_bos_token=True)

    data_args = DataArguments(template="deepseek")
    data_args.cutoff_len = 2000
    data_args.train_on_prompt = False
    data_args.mask_history = False
    data_args.emb_lora = True
    data_args.use_tpe = True
    data_args.path_vocab_path = require_vocab()

    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    ds = json.load(open(require_data("hitab_tpe_train.json")))
    # Pick 2 samples of notably different sizes to exercise padding
    sample_idxs = [0, 244]  # 0 is flat-ish, 244 is the 2744 table (hierarchical)
    examples = {
        "_prompt": [[{"role": "user", "content": ds[i]["prompt"]}] for i in sample_idxs],
        "_response": [[{"role": "assistant", "content": ds[i]["response"]}] for i in sample_idxs],
        "_system": [""] * 2,
        "_tools": [""] * 2,
        "_images": [None] * 2,
        "_videos": [None] * 2,
    }

    out = preprocess_tpe_supervised_dataset(examples, template, tokenizer, None, data_args)

    # Per-sample lengths
    print("Per-sample input_ids lengths (pre-pad):")
    lens = [len(s) for s in out["input_ids"]]
    print(f"  {lens}  (ratio {max(lens)/min(lens):.2f}x)")
    print(f"Per-sample top_path_ids max depth: "
          f"{[max(len(p) for p in s) for s in out['top_path_ids']]}")
    print(f"Per-sample left_path_ids max depth: "
          f"{[max(len(p) for p in s) for s in out['left_path_ids']]}")

    # Build list of features (like HF Trainer does)
    features = []
    for b in range(len(sample_idxs)):
        features.append({
            "input_ids": out["input_ids"][b],
            "attention_mask": [1] * len(out["input_ids"][b]),
            "labels": out["labels"][b],
            "row_ids": out["row_ids"][b],
            "col_ids": out["col_ids"][b],
            "top_path_ids": out["top_path_ids"][b],
            "left_path_ids": out["left_path_ids"][b],
        })

    # First apply TableLoRA collator patch (normally done by load_table_lora)
    load_table_lora()
    # Then chain-patch tpe collator on top
    install_tpe_collator_patch(unk_id=1)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    batch = collator(features)

    print("\n=== Batch tensor shapes ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<20} {type(v).__name__}")

    B, L = batch["input_ids"].shape
    assert batch["row_ids"].shape == (B, L), f"row_ids wrong shape {batch['row_ids'].shape}"
    assert batch["col_ids"].shape == (B, L), f"col_ids wrong shape"
    assert batch["top_path_ids"].dim() == 3, f"top_path_ids must be 3D, got {batch['top_path_ids'].dim()}D"
    assert batch["left_path_ids"].dim() == 3
    assert batch["top_path_ids"].shape[:2] == (B, L)
    assert batch["left_path_ids"].shape[:2] == (B, L)

    # Padding sanity: there should be PATH_PAD (-1) somewhere
    n_pad_top = (batch["top_path_ids"] == PATH_PAD).sum().item()
    n_pad_left = (batch["left_path_ids"] == PATH_PAD).sum().item()
    n_vocab_top = (batch["top_path_ids"] >= 0).sum().item()
    n_vocab_left = (batch["left_path_ids"] >= 0).sum().item()
    print(f"\ntop_path_ids:   PAD({PATH_PAD})={n_pad_top:,}  vocab>=0={n_vocab_top:,}")
    print(f"left_path_ids:  PAD({PATH_PAD})={n_pad_left:,}  vocab>=0={n_vocab_left:,}")
    assert n_pad_top > 0 and n_pad_left > 0, "Expected some -1 padding"

    # Print first few tokens' per-token path ids to eyeball
    print(f"\n=== First 15 non-pad-token path ids (sample 0) ===")
    col_ids0 = batch["col_ids"][0]
    top0 = batch["top_path_ids"][0]
    left0 = batch["left_path_ids"][0]
    attn0 = batch["attention_mask"][0]
    shown = 0
    for t in range(L):
        if attn0[t] == 0:
            continue
        if col_ids0[t] == 0:
            continue
        shown += 1
        top_vals = [int(x) for x in top0[t].tolist() if x != PATH_PAD]
        left_vals = [int(x) for x in left0[t].tolist() if x != PATH_PAD]
        print(f"  pos={t} col_id={col_ids0[t].item():>2}  top_ids={top_vals} left_ids={left_vals}")
        if shown >= 15:
            break

    stats = get_oov_stats()
    print(f"\n=== Collator OOV stats ===")
    print(f"  batches: {stats['batches']}, total path nodes: {stats['total_path_nodes']}, "
          f"unk: {stats['unk_path_nodes']} "
          f"({100*stats['unk_path_nodes']/max(stats['total_path_nodes'],1):.2f}%)")

    print("\n✓ TPE collator smoke test passed.")


if __name__ == "__main__":
    main()
