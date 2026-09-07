"""Evaluate prediction accuracy for HiTab using official hmt_score metric."""
import json
import math
import re
import sys
import os
import unicodedata


# === HiTab official metric (from microsoft/HiTab qa/table/utils.py) ===

def naive_str_to_float(s):
    """Try parsing a string as a number, handling %, $, (), commas."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # Remove $ and %
    s = s.replace("$", "").replace("%", "")
    # Handle parentheses as negative
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # Remove commas (thousands separator)
    s = s.replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


# WikiTableQuestions style normalization
def normalize_string(x):
    """Lowercase, strip, normalize unicode and punctuation."""
    if not isinstance(x, str):
        return x
    # Unicode normalization
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    # Lowercase and strip
    x = x.lower().strip()
    # Replace various dashes with -
    x = re.sub(r"[\u2010-\u2015\u2212]", "-", x)
    # Replace various quotes
    x = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", x)
    x = re.sub(r"[\u201c\u201d\u201e\u201f\u2033]", '"', x)
    # Collapse whitespace
    x = re.sub(r"\s+", " ", x)
    return x


def hmt_process_answer(s):
    """Convert prediction/answer to canonical form."""
    if isinstance(s, list):
        return [hmt_process_answer(x) for x in s]
    if not isinstance(s, str):
        return s
    f = naive_str_to_float(s)
    if f is not None:
        return f
    return normalize_string(s)


def hmt_equal(prediction, answer):
    if type(prediction) != type(answer):
        # Allow int/float interop
        if isinstance(prediction, (int, float)) and isinstance(answer, (int, float)):
            return math.fabs(prediction - answer) < 1e-5
        return False
    if isinstance(prediction, str):
        return prediction == answer
    if isinstance(prediction, (int, float)):
        return math.fabs(prediction - answer) < 1e-5
    if isinstance(prediction, list):
        if len(prediction) != len(answer):
            return False
        return all(hmt_equal(p, a) for p, a in zip(prediction, answer))
    return False


def hmt_score(prediction, answer):
    p = hmt_process_answer(prediction)
    a = hmt_process_answer(answer)
    return hmt_equal(p, a)


def extract_answer(text):
    """Extract the core answer from a possibly verbose model output."""
    text = text.strip()
    # Strip common prefixes
    for prefix in ("the answer is:", "the answer is", "answer:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    # Strip trailing punctuation
    text = text.rstrip(".:;!?").strip()
    return text


def evaluate(predict_dir):
    pred_file = os.path.join(predict_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    correct = 0
    total = 0
    with open(pred_file) as f:
        for line in f:
            item = json.loads(line)
            pred = extract_answer(item.get("predict", ""))
            label = item.get("label", "").strip()
            if hmt_score(pred, label):
                correct += 1
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Results for: {predict_dir}")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_accuracy.py <predict_dir> [predict_dir2 ...]")
        sys.exit(1)

    results = {}
    for predict_dir in sys.argv[1:]:
        acc = evaluate(predict_dir)
        if acc is not None:
            results[predict_dir] = acc

    if len(results) > 1:
        print("\n========== Summary ==========")
        for name, acc in results.items():
            print(f"  {name}: {acc:.2f}%")
