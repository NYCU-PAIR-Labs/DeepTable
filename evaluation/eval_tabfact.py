"""Evaluate TabFact predictions — binary accuracy (True / False / 1 / 0).

Mirrors TabFact's entailment classification metric: percentage of samples
where the predicted label matches the reference after normalisation.

Usage:
    python eval_tabfact.py <predict_dir>
"""
from __future__ import annotations

import json
import os
import re
import sys


TRUE_TOKENS = {"true", "1", "entailed", "yes", "correct", "supported"}
FALSE_TOKENS = {"false", "0", "refuted", "no", "incorrect", "not_entailed",
                "notentailed", "not entailed", "not supported"}


def normalise(text: str) -> str:
    """Map a free-form predict string to {'true','false','other'}."""
    if not isinstance(text, str):
        return "other"
    t = text.strip().lower()
    # Drop common prefixes the model may echo back
    for prefix in ("the answer is:", "the answer is", "answer:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    # Trim trailing punctuation and take first token-ish
    t = re.sub(r"[^\w\s]", " ", t).strip()
    first = t.split()[0] if t else ""
    if first in TRUE_TOKENS:
        return "true"
    if first in FALSE_TOKENS:
        return "false"
    # Last-resort: fuzzy match anywhere in the (trimmed) string
    if any(tok in t for tok in TRUE_TOKENS):
        return "true"
    if any(tok in t for tok in FALSE_TOKENS):
        return "false"
    return "other"


def evaluate(predict_dir: str):
    pred_file = os.path.join(predict_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    correct = 0
    total = 0
    other = 0
    breakdown = {"true": {"total": 0, "correct": 0},
                 "false": {"total": 0, "correct": 0}}

    with open(pred_file) as f:
        for line in f:
            item = json.loads(line)
            p = normalise(item.get("predict", ""))
            r = normalise(item.get("label", ""))
            if r not in ("true", "false"):
                continue  # malformed reference, skip
            total += 1
            breakdown[r]["total"] += 1
            if p == "other":
                other += 1
            elif p == r:
                correct += 1
                breakdown[r]["correct"] += 1

    acc = 100 * correct / total if total else 0.0
    print(f"Results for: {predict_dir}")
    print(f"  N samples:    {total}")
    print(f"  Correct:      {correct}")
    print(f"  Accuracy:     {acc:.2f}%")
    print(f"  Unparseable predictions (other): {other}")
    for k, v in breakdown.items():
        ak = 100 * v["correct"] / v["total"] if v["total"] else 0.0
        print(f"  {k:<5}-labeled: {v['correct']:>5}/{v['total']:<5} = {ak:.2f}%")
    return acc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_tabfact.py <predict_dir>")
        sys.exit(1)

    results = {}
    for d in sys.argv[1:]:
        s = evaluate(d)
        if s is not None:
            results[d] = s

    if len(results) > 1:
        print("\n========== Summary ==========")
        for d, s in results.items():
            print(f"  {d}: Accuracy = {s:.2f}%")
