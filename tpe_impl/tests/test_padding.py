"""Sanity check: padding / pooling correctness.

End-to-end: take 2 train samples, run through tpe_supervised + TPE data collator,
verify:
  1. Padding at sample boundaries uses PATH_PAD (-1)
  2. Token with all-PAD path → pool output = zero vector (not NaN)
  3. Token with partial path → pool output = mean over in-vocab entries
"""
import sys, os, json
from _bootstrap import BACKBONE, require_data, require_gpu, require_vocab  # noqa: F401

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from llamafactory.table_lora.table_lora import load_table_lora

from tpe_impl.processors.tpe_supervised import preprocess_tpe_supervised_dataset
from tpe_impl.processors.metadata_index import reset_for_tests
from tpe_impl.collator import install_tpe_collator_patch, PATH_PAD
from tpe_impl.module.tpe_path_module import TPEPathModule, TPEPathModuleList


MODEL = BACKBONE  # override with $DEEPTABLE_BACKBONE


def main():
    reset_for_tests()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left",
                                        add_eos_token=True, add_bos_token=True)
    data_args = DataArguments(template="deepseek")
    data_args.cutoff_len = 2000
    data_args.train_on_prompt = False
    data_args.mask_history = False
    data_args.emb_lora = True
    data_args.use_tpe = True
    data_args.path_vocab_path = require_vocab()
    template = get_template_and_fix_tokenizer(tok, data_args)

    ds = json.load(open(require_data("hitab_tpe_train.json")))
    examples = {
        "_prompt": [[{"role": "user", "content": ds[i]["prompt"]}] for i in [0, 244]],
        "_response": [[{"role": "assistant", "content": ds[i]["response"]}] for i in [0, 244]],
        "_system": [""] * 2,
        "_tools": [""] * 2,
        "_images": [None] * 2,
        "_videos": [None] * 2,
    }
    out = preprocess_tpe_supervised_dataset(examples, template, tok, None, data_args)
    features = [
        {
            "input_ids": out["input_ids"][b],
            "attention_mask": [1] * len(out["input_ids"][b]),
            "labels": out["labels"][b],
            "row_ids": out["row_ids"][b],
            "col_ids": out["col_ids"][b],
            "top_path_ids": out["top_path_ids"][b],
            "left_path_ids": out["left_path_ids"][b],
        }
        for b in range(2)
    ]
    load_table_lora()
    install_tpe_collator_patch(unk_id=1)
    batch = DataCollatorForSeq2Seq(tokenizer=tok)(features)

    B, L = batch["input_ids"].shape
    top = batch["top_path_ids"]
    left = batch["left_path_ids"]
    attn = batch["attention_mask"]
    col_ids = batch["col_ids"]
    _, _, d_top = top.shape
    _, _, d_left = left.shape

    # (1) Padding at left side (short sample gets padded with -1 paths)
    # Sample 1 was shorter; with padding_side="left" its padded positions are at the FRONT
    sample_1_orig_len = attn[1].sum().item()
    sample_1_pad_len = L - sample_1_orig_len
    print(f"Sample 1: orig_len={sample_1_orig_len}, L={L}, pad_len={sample_1_pad_len}")
    pad_region = top[1, :sample_1_pad_len]  # positions that were padded
    assert torch.all(pad_region == PATH_PAD), \
        f"Expected all -1 in pad region, but got min={pad_region.min()}, max={pad_region.max()}"
    print(f"  ✓ pad region (first {sample_1_pad_len} tokens) all PATH_PAD ({PATH_PAD})")

    # (2) Pool on a random embedding to verify mask correctness
    emb = torch.nn.Embedding(16793, 8)
    emb.weight.data.normal_(0, 0.1)

    # Select a token with all-PAD path (should exist — e.g., prompt text tokens, or padded positions)
    allpad_token = None
    for b in range(B):
        for t in range(L):
            if torch.all(top[b, t] == PATH_PAD):
                allpad_token = (b, t)
                break
        if allpad_token:
            break
    assert allpad_token is not None, "Expected at least one all-PAD token"

    # Select a token with partial path (some real + some pad)
    partial_token = None
    for b in range(B):
        for t in range(L):
            n_pad = int((top[b, t] == PATH_PAD).sum())
            n_valid = d_top - n_pad
            if 0 < n_valid < d_top:
                partial_token = (b, t, n_valid)
                break
        if partial_token:
            break

    # Select a token with full path (all valid)
    full_token = None
    for b in range(B):
        for t in range(L):
            if torch.all(top[b, t] >= 0):
                full_token = (b, t)
                break
        if full_token:
            break

    pooled = TPEPathModule.pool(top, emb)
    assert pooled.shape == (B, L, 8)
    # Check all-PAD token → zero vector
    z = pooled[allpad_token[0], allpad_token[1]]
    assert torch.all(z == 0), f"all-PAD pool should be 0, got {z}"
    assert not torch.any(torch.isnan(z)), f"all-PAD pool should not be NaN"
    print(f"  ✓ all-PAD token (b={allpad_token[0]}, t={allpad_token[1]}) → pool output = 0")

    if partial_token:
        b, t, n_valid = partial_token
        path = top[b, t]
        valid_ids = path[path >= 0]
        expected = emb(valid_ids).mean(dim=0)
        assert torch.allclose(pooled[b, t], expected, atol=1e-6), \
            f"partial pool mismatch at (b={b}, t={t})"
        print(f"  ✓ partial ({n_valid}/{d_top} valid) token → pool = mean(valid)")

    if full_token:
        b, t = full_token
        path = top[b, t]
        expected = emb(path).mean(dim=0)
        assert torch.allclose(pooled[b, t], expected, atol=1e-6)
        print(f"  ✓ full-path ({d_top}/{d_top} valid) token → pool = mean(all)")

    print("\n✓ Sanity 3 padding / pooling passed.")


if __name__ == "__main__":
    main()
