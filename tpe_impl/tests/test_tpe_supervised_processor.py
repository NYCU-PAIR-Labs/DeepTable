"""Smoke test: run tpe_supervised on one sample from hitab_tpe_train,
print resulting features, verify token-level path alignment."""
import json, sys, os
from pathlib import Path

from _bootstrap import BACKBONE, require_data, require_gpu, require_vocab  # noqa: F401

from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

from tpe_impl.processors.tpe_supervised import preprocess_tpe_supervised_dataset
from tpe_impl.processors.metadata_index import reset_for_tests

MODEL = BACKBONE  # override with $DEEPTABLE_BACKBONE


def main():
    reset_for_tests()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left",
                                              add_eos_token=True, add_bos_token=True)

    # Use the real DataArguments dataclass — patch fields we care about
    from dataclasses import asdict
    data_args = DataArguments(template="deepseek")
    data_args.cutoff_len = 2000
    data_args.train_on_prompt = False
    data_args.mask_history = False
    # Extend with tpe fields (not part of official DataArguments)
    data_args.emb_lora = True
    data_args.use_tpe = True
    data_args.path_vocab_path = require_vocab()

    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load one sample from hitab_tpe_train
    ds = json.load(open(require_data("hitab_tpe_train.json")))
    sample = ds[0]
    print(f"Sample 0 prompt length (chars): {len(sample['prompt'])}")
    print(f"Sample 0 response: {sample['response']}")
    print()

    # Mimic examples dict that LLaMA-Factory produces (post-aligner shape)
    examples = {
        "_prompt": [[{"role": "user", "content": sample["prompt"]}]],
        "_response": [[{"role": "assistant", "content": sample["response"]}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
    }

    # Run
    out = preprocess_tpe_supervised_dataset(examples, template, tokenizer, None, data_args)

    # Report
    n = len(out["input_ids"][0])
    print(f"Output features for 1 sample:")
    print(f"  input_ids:      len={n}")
    print(f"  row_ids:        len={len(out['row_ids'][0])}")
    print(f"  col_ids:        len={len(out['col_ids'][0])}")
    print(f"  top_path_ids:   len={len(out['top_path_ids'][0])}  "
          f"(list-of-list, depths: min={min(len(p) for p in out['top_path_ids'][0])}, "
          f"max={max(len(p) for p in out['top_path_ids'][0])})")
    print(f"  left_path_ids:  len={len(out['left_path_ids'][0])}  "
          f"(depths: min={min(len(p) for p in out['left_path_ids'][0])}, "
          f"max={max(len(p) for p in out['left_path_ids'][0])})")

    # Alignment spot check: find first table token (col_id > 0) and print
    row_ids = out["row_ids"][0]
    col_ids = out["col_ids"][0]
    tops = out["top_path_ids"][0]
    lefts = out["left_path_ids"][0]
    input_ids = out["input_ids"][0]

    # Decode vocabulary back for readable report
    vocab_meta = json.load(open(data_args.path_vocab_path))
    id_to_tok = vocab_meta["vocab"]

    def paths_to_str(pids):
        if not pids:
            return "[]"
        return "[" + ",".join(id_to_tok[x] if x < len(id_to_tok) else f"UNK#{x}" for x in pids) + "]"

    # Print first 40 table tokens with full context
    print(f"\n=== First 40 table tokens (col_id > 0) ===")
    print(f"{'pos':>4} {'row':>3} {'col':>3}  {'token':<20} top_path / left_path")
    printed = 0
    for t in range(n):
        if col_ids[t] == 0:
            continue
        tok_str = tokenizer.decode([input_ids[t]]).replace("\n", "\\n")
        print(f"{t:>4} {row_ids[t]:>3} {col_ids[t]:>3}  {tok_str!r:<22}  "
              f"top={paths_to_str(tops[t])}  left={paths_to_str(lefts[t])}")
        printed += 1
        if printed >= 40:
            print(f"  ... ({sum(1 for c in col_ids if c > 0) - 40} more table tokens)")
            break

    # Invariants
    assert all(isinstance(p, list) for p in tops), "top_path_ids entries must be lists"
    assert all(isinstance(p, list) for p in lefts), "left_path_ids entries must be lists"
    assert all(
        (col_ids[t] == 0) == (tops[t] == [] and lefts[t] == [])
        or col_ids[t] > 0
        for t in range(n)
    ), "col_id=0 tokens should have empty paths"

    # OOV stats
    from tpe_impl.processors.tpe_supervised import get_oov_stats
    stats = get_oov_stats()
    print(f"\n=== OOV stats after 1 sample ===")
    print(f"  total path nodes: {stats['total_path_nodes']}")
    print(f"  unk path nodes:   {stats['unk_path_nodes']}")
    print(f"  samples processed: {stats['samples_processed']}")

    print("\n✓ TPE supervised-processor smoke test passed.")


if __name__ == "__main__":
    main()
