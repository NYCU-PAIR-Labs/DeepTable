# Training configurations

Four YAMLs covering the four rows of Table 2, on HiTab with the
DeepSeek-LLM-7B-Chat backbone. Every hyperparameter matches Appendix G
(Table: *Training hyperparameters*) — these are the values used for the
reported results, not placeholders.

| File | Row in Table 2 | Launch with |
|---|---|---|
| `hitab_tablelora.yaml` | TableLoRA baseline | `llamafactory-cli train configs/hitab_tablelora.yaml` |
| `hitab_sab.yaml` | SAB only | `SAB_LR_MULTIPLIER=1000 llamafactory-cli train configs/hitab_sab.yaml` |
| `hitab_tpe.yaml` | TPE only | `python -m tpe_impl.bin.run_tpe_training configs/hitab_tpe.yaml` |
| `hitab_sab_tpe.yaml` | Full (SAB + TPE) | `SAB_LR_MULTIPLIER=1000 python -m tpe_impl.bin.run_tpe_training configs/hitab_sab_tpe.yaml` |

## Hyperparameters (Appendix G)

| | Value |
|---|---|
| LoRA / 2D-LoRA rank | 8 |
| Adapted modules | key and value projections (`k_proj,v_proj`) |
| LoRA alpha / dropout | 16 / 0.1 |
| Base learning rate | 5e-6, cosine, no warmup |
| Add-on LR multiplier λ | **1000**, shared across backbones and datasets |
| Epochs | 3 (all datasets) |
| Serialization cutoff | 2000 (4000 for TabFact) |
| Effective batch size | 16 |
| Precision | bf16 |
| Seeds | 0, 1, 2, 3 |
| Decoding | greedy |

λ is applied to SAB via the `SAB_LR_MULTIPLIER` **environment variable** and to
TPE via the `tpe_lr_multiplier` **YAML key**. Both modules have very few
parameters — 1,920 scalars for SAB on DeepSeek (30 layers × 32 heads × 2) —
so the base learning rate barely moves them. Forgetting `SAB_LR_MULTIPLIER`
runs SAB at 1× and will not reproduce the paper's numbers.

Reproduce all four seeds by setting `seed:` to 0, 1, 2, 3 in turn (and giving
each its own `output_dir`).

## Switching benchmark

All four conditions read the **same** dataset for a given benchmark — the
per-cell-metadata file. For the non-TPE rows LLaMA-Factory's aligner simply
drops the extra metadata column, so one dataset covers every row of Table 2.

| Benchmark | `dataset` | `cutoff_len` | Batch split |
|---|---|---|---|
| HiTab   | `hitab_tpe_train`      | 2000 | 2 × 8 |
| WikiTQ  | `wikitq_tpe_v2_train`  | 2000 | 2 × 8 |
| FeTaQA  | `fetaqa_tpe_v2_train`  | 2000 | 2 × 8 |
| TabFact | `tabfact_tpe_v2_train` | 4000 | 1 × 16 |

TabFact's longer cutoff needs `per_device_train_batch_size: 1` with
`gradient_accumulation_steps: 16` to fit in 48 GB; the effective batch size is
still 16. Its `logging_steps` was 50 rather than 10.

The `*_tpe_v2_*` files are what the `..._leftpath_variant.py` preprocessing
scripts produce — see README step 2. Only HiTab has a genuine row-header
hierarchy, so the other three use the synthesized-`left_path` variant.

**Path vocabulary.** HiTab-only runs use `hitab_tpe_path_vocab.json`
(16,793 nodes, per Appendix G). Runs on any other benchmark must point
`path_vocab_path` at `combined_path_vocab.json` instead.

## Other backbones

Appendix G reports Qwen2.5-7B-Instruct and Llama-3-8B-Instruct alongside
DeepSeek. They share this training configuration exactly; only
`model_name_or_path` and `template` change (`qwen` / `llama3`). SAB's parameter
count follows from the architecture: 1,568 for Qwen2.5 (28 × 28 × 2), 2,048 for
Llama-3 (32 × 32 × 2). Both use grouped-query attention, where the structural
bias is added after `repeat_kv` with an independent bias per query head.

## Predicting

Copy the training YAML, then set `do_train: false`, `do_predict: true`,
`adapter_name_or_path: <output_dir>`, and point `dataset` at the matching
`*_test` split. Decoding is greedy. Launch with the same command as training —
TPE runs must still go through `run_tpe_training.py`, which is what reloads
`tpe_modules.safetensors` from the adapter directory.
