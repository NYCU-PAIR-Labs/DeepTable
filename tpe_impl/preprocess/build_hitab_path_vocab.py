"""Build path vocabulary from hitab_tpe_train.json.

Vocabulary layout:
  id 0: <PAD_PATH>     reserved; padding ids in collator use -1 (not 0) but 0
                       reserved anyway to avoid "forgot to offset" bugs.
  id 1: <UNK_PATH>     unseen node values at test/inference time.
  id 2+: distinct node values from train set top_path ∪ left_path.

Output ($DEEPTABLE_DATA_DIR, default <repo>/LLaMA-Factory/data):
  hitab_tpe_path_vocab.json
    {
      "pad_id": 0,
      "unk_id": 1,
      "vocab": ["<PAD_PATH>", "<UNK_PATH>", "<node_value_1>", ...],
      "token_to_id": {"<PAD_PATH>": 0, "<UNK_PATH>": 1, ...}
    }

Also reports test-set unknown rate (fraction of node value occurrences in test
set whose value string is not in train vocab).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deeptable_paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR
TRAIN = DATA / "hitab_tpe_train.json"
TEST = DATA / "hitab_tpe_test.json"
VOCAB_OUT = DATA / "hitab_tpe_path_vocab.json"

PAD_TOKEN = "<PAD_PATH>"
UNK_TOKEN = "<UNK_PATH>"


def collect_node_values(dataset) -> Counter:
    """Walk all per_cell_metadata, return Counter of every node value string appearance."""
    ctr: Counter = Counter()
    for sample in dataset:
        md = json.loads(sample["per_cell_metadata_json"])
        for cell in md:
            for v in cell.get("top_path", []):
                ctr[v] += 1
            for v in cell.get("left_path", []):
                ctr[v] += 1
    return ctr


def main():
    print("Loading datasets...", flush=True)
    train_ds = json.load(TRAIN.open())
    test_ds = json.load(TEST.open())

    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}", flush=True)

    # Collect train vocab
    train_ctr = collect_node_values(train_ds)
    print(f"Train: {sum(train_ctr.values()):,} path-node occurrences / "
          f"{len(train_ctr):,} unique values", flush=True)

    # Vocabulary: deterministic order (frequency desc, then alphabetical)
    vocab = [PAD_TOKEN, UNK_TOKEN] + [
        v for v, _ in sorted(train_ctr.items(), key=lambda x: (-x[1], x[0]))
    ]
    token_to_id = {v: i for i, v in enumerate(vocab)}
    print(f"Vocab size (incl PAD + UNK): {len(vocab)}", flush=True)

    # Test-set unknown rate
    test_ctr = collect_node_values(test_ds)
    test_total = sum(test_ctr.values())
    test_unk_occurrences = sum(
        count for v, count in test_ctr.items() if v not in token_to_id
    )
    test_unk_unique = sum(
        1 for v in test_ctr if v not in token_to_id
    )
    print(f"Test: {test_total:,} occurrences / {len(test_ctr):,} unique values", flush=True)
    print(f"Test unknown (occurrence): {test_unk_occurrences:,} / {test_total:,} "
          f"({100*test_unk_occurrences/max(test_total,1):.2f}%)", flush=True)
    print(f"Test unknown (unique): {test_unk_unique:,} / {len(test_ctr):,} "
          f"({100*test_unk_unique/max(len(test_ctr),1):.2f}%)", flush=True)

    # Save vocabulary
    meta = {
        "pad_id": 0,
        "unk_id": 1,
        "vocab": vocab,
        "token_to_id": token_to_id,
        "train_occurrences": sum(train_ctr.values()),
        "train_unique_values": len(train_ctr),
        "test_unk_occurrence_rate": test_unk_occurrences / max(test_total, 1),
        "test_unk_unique_rate": test_unk_unique / max(len(test_ctr), 1),
    }
    VOCAB_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {VOCAB_OUT} ({VOCAB_OUT.stat().st_size:,} bytes)")

    # Report: top 20 + bottom 20 + unknown samples
    print("\n=== Top 20 most-frequent path nodes (train) ===")
    for v, c in sorted(train_ctr.items(), key=lambda x: -x[1])[:20]:
        print(f"  {c:>8,} × {v!r}")

    print("\n=== 20 random rarest path nodes (train, count=1) ===")
    import random
    singletons = [v for v, c in train_ctr.items() if c == 1]
    print(f"  (total {len(singletons)} singletons)")
    rng = random.Random(42)
    for v in rng.sample(singletons, min(20, len(singletons))):
        print(f"  1 × {v!r}")

    print("\n=== Up to 20 test-only (unknown) node values ===")
    unk_in_test = [v for v in test_ctr if v not in token_to_id]
    print(f"  (total {len(unk_in_test)} unique unknowns in test)")
    for v in (rng.sample(unk_in_test, 20) if len(unk_in_test) >= 20 else unk_in_test):
        count = test_ctr[v]
        print(f"  {count} × {v!r}")


if __name__ == "__main__":
    main()
