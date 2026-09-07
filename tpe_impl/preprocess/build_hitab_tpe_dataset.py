"""Build hitab_tpe_train.json / hitab_tpe_test.json from raw HiTab data.

Combines each raw HiTab sample (question + answer + table_id) with a hmt-rendered
prompt + per-cell metadata produced by `render_from_hmt.build_prompt`.

Input (all under $DEEPTABLE_HITAB_DIR, default <repo>/table_preprocess/hitab):
  - train: train_samples.jsonl              (from microsoft/HiTab)
  - test:  hitab_test.json                  (from osunlp/TableInstruct)
  - hmt:   hmt/{table_id}.json              (from microsoft/HiTab)

Output (under $DEEPTABLE_DATA_DIR, default <repo>/LLaMA-Factory/data):
  - hitab_tpe_train.json
  - hitab_tpe_test.json

Schema (per sample):
  {
      "prompt": str,
      "response": str,
      "per_cell_metadata_json": str      # json.dumps(list_of_cell_dicts)
  }

Per spec, `per_cell_metadata_json` is a serialized JSON STRING (not a nested
column) to avoid HF Arrow schema headaches across vocabulary changes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from render_from_hmt import build_prompt  # noqa: E402
from deeptable_paths import DATA_DIR, HITAB_DIR  # noqa: E402

HITAB_ROOT = HITAB_DIR
HMT_DIR = HITAB_ROOT / "hmt"
TRAIN_JSONL = HITAB_ROOT / "train_samples.jsonl"
TEST_JSON = HITAB_ROOT / "hitab_test.json"
OUT_DIR = DATA_DIR


def load_hmt(table_id: str) -> Dict[str, Any]:
    with open(HMT_DIR / f"{table_id}.json") as f:
        return json.load(f)


def format_answer(answer) -> str:
    """Mirror TableLoRA's answer formatting for consistency with existing test set."""
    if isinstance(answer, list):
        if len(answer) == 1:
            return str(answer[0])
        return ", ".join(f"<{a}>" for a in answer)
    return str(answer)


def process_train():
    out = []
    skipped = []
    with open(TRAIN_JSONL) as f:
        samples = [json.loads(line) for line in f]
    for s in tqdm(samples, desc="train"):
        try:
            hmt = load_hmt(s["table_id"])
            prompt, meta = build_prompt(hmt, s["question"])
        except FileNotFoundError:
            skipped.append((s["id"], s["table_id"], "hmt missing"))
            continue
        except AssertionError as e:
            skipped.append((s["id"], s["table_id"], f"assert: {e}"))
            continue
        except Exception as e:
            skipped.append((s["id"], s["table_id"], f"{type(e).__name__}: {e}"))
            continue
        out.append({
            "prompt": prompt,
            "response": format_answer(s["answer"]),
            "per_cell_metadata_json": json.dumps(meta, ensure_ascii=False),
        })
    outp = OUT_DIR / "hitab_tpe_train.json"
    with open(outp, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[build_dataset] train: wrote {len(out)} / skipped {len(skipped)} → {outp}")
    if skipped:
        print(f"  first 5 skipped: {skipped[:5]}")
    return out, skipped


def process_test():
    out = []
    skipped = []
    with open(TEST_JSON) as f:
        samples = json.load(f)
    for s in tqdm(samples, desc="test"):
        try:
            hmt = load_hmt(s["table_id"])
            prompt, meta = build_prompt(hmt, s["question"])
        except FileNotFoundError:
            skipped.append((s["table_id"], "hmt missing"))
            continue
        except AssertionError as e:
            skipped.append((s["table_id"], f"assert: {e}"))
            continue
        except Exception as e:
            skipped.append((s["table_id"], f"{type(e).__name__}: {e}"))
            continue
        # TableInstruct test schema: `output` is the answer string
        resp = s.get("output", "")
        out.append({
            "prompt": prompt,
            "response": resp,
            "per_cell_metadata_json": json.dumps(meta, ensure_ascii=False),
        })
    outp = OUT_DIR / "hitab_tpe_test.json"
    with open(outp, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[build_dataset] test: wrote {len(out)} / skipped {len(skipped)} → {outp}")
    if skipped:
        print(f"  first 5 skipped: {skipped[:5]}")
    return out, skipped


if __name__ == "__main__":
    train_out, train_skip = process_train()
    test_out, test_skip = process_test()

    # Quick schema report
    print()
    print(f"Train: {len(train_out)} samples, {len(train_skip)} skipped")
    print(f"Test:  {len(test_out)} samples, {len(test_skip)} skipped")
    if train_out:
        print("\nExample prompt head (train[0]):")
        print(train_out[0]["prompt"][:400])
        print("...")
        print("response:", train_out[0]["response"])
        md = json.loads(train_out[0]["per_cell_metadata_json"])
        print(f"metadata: {len(md)} cells, first 3:")
        for m in md[:3]:
            print(f"  {m}")
