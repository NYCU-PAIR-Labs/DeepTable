"""Evaluate FeTaQA predictions with corpus BLEU-4 (sacrebleu).

Mirrors the FeTaQA paper's evaluation: corpus-level BLEU-4 on the generated
sentence against the reference answer.

Usage:
    python eval_fetaqa_bleu.py <predict_dir>

Where `<predict_dir>/generated_predictions.jsonl` has one JSON object per line
with keys `predict` (model output) and `label` (reference answer).
"""
from __future__ import annotations

import json
import os
import re
import sys

import sacrebleu


def extract_answer(text: str) -> str:
    """Normalise model output:
      - strip whitespace
      - remove common prefixes the model may have echoed
      - keep everything else (do NOT truncate — FeTaQA references are full sentences)
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    for prefix in ("the answer is:", "the answer is", "answer:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    return t


def evaluate(predict_dir: str):
    pred_file = os.path.join(predict_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    preds, refs = [], []
    with open(pred_file) as f:
        for line in f:
            item = json.loads(line)
            p = extract_answer(item.get("predict", ""))
            r = (item.get("label") or "").strip()
            if not r:  # skip empty references
                continue
            preds.append(p)
            refs.append(r)

    # sacrebleu expects list of predictions and list-of-lists of references
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    # Sample-level BLEU-1..4 for reference (not the primary metric)
    print(f"Results for: {predict_dir}")
    print(f"  N samples: {len(preds)}")
    print(f"  corpus BLEU-4: {bleu.score:.2f}")
    print(f"  precisions (1-gram, 2-gram, 3-gram, 4-gram): "
          f"{[f'{p:.1f}' for p in bleu.precisions]}")
    print(f"  length ratio (sys/ref): {bleu.sys_len}/{bleu.ref_len}")
    print(f"  brevity penalty: {bleu.bp:.3f}")
    return bleu.score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_fetaqa_bleu.py <predict_dir>")
        sys.exit(1)

    results = {}
    for d in sys.argv[1:]:
        s = evaluate(d)
        if s is not None:
            results[d] = s

    if len(results) > 1:
        print("\n========== Summary ==========")
        for d, s in results.items():
            print(f"  {d}: BLEU-4 = {s:.2f}")
