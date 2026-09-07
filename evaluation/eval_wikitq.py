"""WikiTQ-style evaluator.

The Stanford WikiTableQuestions official evaluator does:
  1. Parse the gold label into a LIST of target values (in TSV, split by tab;
     in our TableLoRA-joined format, list is joined by ',').
  2. Normalise each token (number-aware + string normalize).
  3. Compare prediction-vs-gold as UNORDERED SETS.

Our `eval_accuracy.py` (HiTab's `hmt_score`) compares as ORDERED lists and uses
',' as a mid-string character, so true multi-value WTQ answers get marked wrong
on trivial reordering. This script fixes that.

Strategy to handle thousand-separator ambiguity ("100,000" vs multi-value):
  - If the gold parses as a single number (strip `,` and try float),
    treat as SINGLE-VALUE NUMBER.
  - Else if gold contains ',', split by ',' and set-compare tokens.
  - Else single-value string normalise.

Usage:
    python evaluation/eval_wikitq.py <predict_dir> [more_dirs...] [more_dirs...]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata


# --------- primitive normalisers (copied from eval_accuracy.py) ---------

def naive_str_to_float(s):
    if not isinstance(s, str): return None
    s = s.strip()
    if not s: return None
    s = s.replace("$", "").replace("%", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def normalize_string(x):
    if not isinstance(x, str): return x
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower().strip()
    x = re.sub(r"[\u2010-\u2015\u2212]", "-", x)
    x = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", x)
    x = re.sub(r"[\u201c\u201d\u201e\u201f\u2033]", '"', x)
    x = re.sub(r"\s+", " ", x)
    return x


def process_token(tok):
    """Try number first; fall back to normalised string."""
    f = naive_str_to_float(tok)
    if f is not None:
        return ("num", f)
    return ("str", normalize_string(tok))


def tokens_equal(a, b):
    """(kind, val) equality. Numbers compared within 1e-5."""
    if a[0] != b[0]:
        # int vs float both fit in num kind; other cross-kind is unequal
        return False
    if a[0] == "num":
        return math.fabs(a[1] - b[1]) < 1e-5
    return a[1] == b[1]


# --------- WTQ-style comparison ---------

_THOUSAND_SEP_NUM = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def parse_label(label):
    """Return set of (kind, value) tokens.

    Rules:
      - If label matches a proper thousand-separator number pattern
        ("100,000" or "1,234,567" or "1,234.56"), treat as single number.
      - Else if label has AT MOST one comma and parses naively as number
        ($, %, (neg), single comma allowed), treat as single number.
      - Else if label contains ',', split and set-compare as WTQ official does.
      - Else single string.
    """
    if not isinstance(label, str):
        return set()
    label = label.strip()
    if not label:
        return set()
    # (a) proper thousand-separator number pattern (handles "1,234,567")
    if _THOUSAND_SEP_NUM.match(label):
        return {("num", float(label.replace(",", "")))}
    # (b) naive number parse — but only if at most one comma
    if label.count(",") <= 1:
        f = naive_str_to_float(label)
        if f is not None:
            return {("num", f)}
    # (c) multi-value: split by comma, set-compare
    if "," in label:
        parts = [p.strip() for p in label.split(",")]
        return {process_token(p) for p in parts if p}
    return {process_token(label)}


def extract_answer(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.strip()
    for prefix in ("the answer is:", "the answer is", "answer:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    t = t.rstrip(".:;!?")
    return t.strip()


def evaluate(predict_dir):
    pred_file = os.path.join(predict_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    correct = 0
    total = 0
    multi_val_total = 0
    multi_val_correct = 0
    with open(pred_file) as f:
        for line in f:
            item = json.loads(line)
            pred_raw = extract_answer(item.get("predict", ""))
            gold_raw = (item.get("label") or "").strip()
            if not gold_raw:
                continue
            gold_set = parse_label(gold_raw)
            pred_set = parse_label(pred_raw)
            ok = gold_set == pred_set
            is_multi = "," in gold_raw and len(gold_set) > 1
            total += 1
            if is_multi:
                multi_val_total += 1
            if ok:
                correct += 1
                if is_multi:
                    multi_val_correct += 1

    acc = 100 * correct / total if total else 0.0
    mv_acc = 100 * multi_val_correct / multi_val_total if multi_val_total else 0.0
    print(f"Results for: {predict_dir}")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  (multi-value subset: {multi_val_correct}/{multi_val_total} = {mv_acc:.2f}%)")
    return acc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_wikitq.py <predict_dir> [...]")
        sys.exit(1)

    results = {}
    for d in sys.argv[1:]:
        s = evaluate(d)
        if s is not None:
            results[d] = s

    if len(results) > 1:
        print("\n========== Summary ==========")
        for d, s in results.items():
            print(f"  {d}: {s:.2f}%")
