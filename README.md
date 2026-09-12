# DeepTable

Reference implementation for **"DeepTable: Structural Attention Biases and Tree
Path Encoding for Hierarchical Table Understanding."** It implements the two
proposed modules — SAB (Section 3.3) and TPE (Section 3.4) — and their
integration into the TableLoRA / LLaMA-Factory training pipeline (Appendix H).

**Paper:** [arXiv:2609.07707](https://arxiv.org/abs/2609.07707)

```bash
bash setup.sh                                  # LLaMA-Factory v0.9.1 + overlay + HiTab
# ... preprocess (steps 1-2 below) ...
python -m tpe_impl.bin.run_tpe_training configs/hitab_sab_tpe.yaml
```

See [`configs/`](configs/) for the four training configurations, and
["How to reproduce"](#how-to-reproduce) for the full pipeline.

Licensed under the MIT License; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)
for the third-party code vendored here and its licensing.

---

## Results

Four-seed greedy means, reproduced by the pipeline below. **Bold** is the best
result per column within a backbone; parentheses give the delta against that
backbone's TableLoRA baseline. Trained checkpoints for every cell in this
table are on the Hugging Face Hub — see "Trained checkpoints" further down.

| Backbone | Method | SAB | TPE | HiTab (acc) | WikiTQ (acc) | FeTaQA (BLEU) | TabFact (acc) |
|---|---|:-:|:-:|---|---|---|---|
| **DeepSeek-LLM-7B-Chat** | TableLoRA | – | – | 46.94 | 40.42 | 27.29 | **77.05** |
| | DeepTable w/o TPE | ✓ | ✗ | **52.05** (+5.11) | 42.79 (+2.37) | 29.52 (+2.23) | 76.35 (−0.70) |
| | DeepTable w/o SAB | ✗ | ✓ | 45.83 (−1.11) | 42.30 (+1.88) | **30.32** (+3.03) | 74.44 (−2.61) |
| | DeepTable (full) | ✓ | ✓ | 51.20 (+4.26) | **43.08** (+2.66) | 30.27 (+2.98) | 75.03 (−2.02) |
| **Llama-3-8B-Instruct** | TableLoRA | – | – | 58.56 | 53.45 | 30.23 | 84.01 |
| | DeepTable w/o TPE | ✓ | ✗ | **74.01** (+15.45) | **59.81** (+6.36) | 32.63 (+2.40) | **85.06** (+1.05) |
| | DeepTable w/o SAB | ✗ | ✓ | 71.83 (+13.27) | 59.19 (+5.74) | 32.66 (+2.43) | 84.40 (+0.39) |
| | DeepTable (full) | ✓ | ✓ | 73.39 (+14.83) | 59.45 (+6.00) | **32.79** (+2.56) | 84.36 (+0.35) |
| **Qwen2.5-7B-Instruct** | TableLoRA | – | – | 62.38 | 52.95 | 30.49 | 81.20 |
| | DeepTable w/o TPE | ✓ | ✗ | 65.36 (+2.98) | 53.79 (+0.84) | 30.96 (+0.47) | **81.33** (+0.13) |
| | DeepTable w/o SAB | ✗ | ✓ | 62.34 (−0.04) | 53.21 (+0.26) | 30.77 (+0.28) | 80.71 (−0.49) |
| | DeepTable (full) | ✓ | ✓ | **65.56** (+3.18) | **53.99** (+1.04) | **30.97** (+0.48) | 81.14 (−0.06) |

Which config produces which row:

| Row | Config |
|---|---|
| TableLoRA | `configs/hitab_tablelora.yaml` |
| DeepTable w/o TPE (SAB only) | `configs/hitab_sab.yaml` |
| DeepTable w/o SAB (TPE only) | `configs/hitab_tpe.yaml` |
| DeepTable (full) | `configs/hitab_sab_tpe.yaml` |

> **Where the TableLoRA baselines come from.** For DeepSeek-LLM-7B-Chat and
> Llama-3-8B-Instruct, the TableLoRA numbers are taken from
> [He et al. (2025)](https://github.com/microsoft/TableLoRA) rather than re-run
> here. Only the Qwen2.5-7B baseline was reproduced locally, since that backbone
> was not evaluated in the original paper. If you train
> `hitab_tablelora.yaml` yourself, expect your number to land near — but not
> exactly on — the published DeepSeek/Llama-3 baseline.

**Checking your own run.** The per-seed HiTab accuracies for
DeepSeek-LLM-7B-Chat are a convenient anchor, since a single seed is far
cheaper to reproduce than all four:

| Config | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| DeepTable w/o TPE | 53.79 | 51.58 | 51.20 | 51.64 | 52.05 |
| DeepTable (full) | 53.28 | 51.01 | 50.44 | 50.06 | 51.20 |

Per-seed results for every backbone and benchmark, along with significance
tests, are in the paper's appendix.

**Naming.** Throughout this codebase, "TPE" always refers to Tree Path Encoding
(Section 3.4). An earlier internal codename ("Hier-LoRA" / "hier") has been
renamed to TPE everywhere in this package — class names, function names, config
fields, environment variables, and file names — so the two names in the paper,
SAB and TPE, are the only two module names used here.

**Provenance.** Three different codebases are combined here:
- Code we wrote for this paper: `tpe_impl/`, `llamafactory_src_overlay/llamafactory/sab/`,
  and the row/col-id and SAB/TPE wiring inside the other `llamafactory_src_overlay/`
  files (marked below).
- Code vendored from the official **TableLoRA** release (He et al., ACL 2025;
  Copyright Microsoft Corporation, MIT License), which we extend: `table_preprocess/`,
  `llamafactory_src_overlay/llamafactory/table_lora/`. Included only for context —
  not our contribution.
- Code vendored from **LLaMA-Factory** v0.9.1 (Copyright the LlamaFactory team,
  Apache-2.0 License), patched in a few places to plumb SAB/TPE through the
  training pipeline: the remaining files under `llamafactory_src_overlay/llamafactory/`.

**Paths.** Every filesystem location the pipeline touches is resolved in
[`deeptable_paths.py`](deeptable_paths.py), from environment variables with
repo-relative defaults matching what `setup.sh` produces. Nothing is hard-coded
to a particular machine. Override any of these to relocate things:

| Variable | Default | What lives there |
|---|---|---|
| `DEEPTABLE_LF_ROOT` | `<repo>/LLaMA-Factory` | The LLaMA-Factory v0.9.1 checkout `setup.sh` clones |
| `DEEPTABLE_DATA_DIR` | `$DEEPTABLE_LF_ROOT/data` | Preprocessed datasets + path vocabularies |
| `DEEPTABLE_HITAB_DIR` | `<repo>/table_preprocess/hitab` | Raw HiTab files (`train_samples.jsonl`, `hitab_test.json`, `hmt/`) |
| `DEEPTABLE_OUTPUT_DIR` | `$DEEPTABLE_LF_ROOT/output` | Training outputs / adapter checkpoints |

If you set one for `setup.sh`, keep it exported for every later command —
`DEEPTABLE_DATA_DIR` in particular must agree with LLaMA-Factory's own `data/`
directory, since that is where it resolves `dataset_info.json` entries from.

**Hyperparameters.** LoRA rank, learning rate, epochs, and the SAB/TPE
learning-rate multipliers are in Appendix G of the paper. The YAMLs in
`configs/` carry runnable defaults with the Appendix-G-swept values marked as
such; check the paper before reproducing a specific number. Hyperparameter-sweep
configs and training logs are not included here.

**Trained checkpoints.** Every adapter behind Table 2 is published on the
Hugging Face Hub — SAB-only, TPE-only, and full DeepTable, across all four
benchmarks and four seeds per backbone:

| Backbone | Checkpoints |
|---|---|
| DeepSeek-LLM-7B-Chat | [e54true/deeptable-checkpoints-deepseek7b](https://huggingface.co/e54true/deeptable-checkpoints-deepseek7b) |
| Llama-3-8B-Instruct | [e54true/deeptable-checkpoints-llama3-8b](https://huggingface.co/e54true/deeptable-checkpoints-llama3-8b) |
| Qwen2.5-7B-Instruct | [e54true/deeptable-checkpoints-qwen25-7b](https://huggingface.co/e54true/deeptable-checkpoints-qwen25-7b) |

TableLoRA baseline checkpoints are **only** included for Qwen2.5 — that is the
one backbone where the paper's Table 2 baseline number was reproduced locally
rather than cited from [He et al. (2025)](https://github.com/microsoft/TableLoRA);
see each repo's model card for the reasoning and the verification against the
paper's numbers. Every checkpoint packs two stacked PEFT adapters (TableLoRA's
prompt encoder plus the actual 2D-LoRA weights) — see a model card's "How to
load" section before using one directly.

**Note.** This is a research codebase, not a packaged library.

---

## How to reproduce

**0. Environment.**
```bash
bash setup.sh   # clones LLaMA-Factory v0.9.1, applies the overlay, installs
                # requirements.txt, downloads the raw HiTab files
```
This installs everything relative to the repo (see the **Paths** table above);
set `DEEPTABLE_LF_ROOT` first if you want LLaMA-Factory elsewhere. WikiTQ,
FeTaQA and TabFact are pulled straight from the HuggingFace `datasets` library
by `table_preprocess/{wikitq,fetaqa,tabfact}.py`, so only HiTab needs its own
download.

`setup.sh` also performs step 3 (registering the datasets and installing the
overlay), so you do not need to repeat it by hand.

**1. Base table serialization** (Section 3.2).

TPE's preprocessing (step 2) needs two kinds of input, and only one of them has
to be generated:

*FeTaQA, TabFact and WikiTQ* — use the data files that ship with the official
[TableLoRA release](https://github.com/microsoft/TableLoRA), `data/{fetaqa,
tabfact,wikitq}_{train,test}.json`. These are exactly the inputs used for the
paper's reported runs (verified by md5), already in the `<COL>`/`<ROW>` format
that step 2 parses. Copy them into `$DEEPTABLE_DATA_DIR` and you are done:

```bash
git clone --depth 1 https://github.com/microsoft/TableLoRA.git /tmp/tablelora
cp /tmp/tablelora/data/{fetaqa,tabfact,wikitq}_{train,test}.json \
   "${DEEPTABLE_DATA_DIR:-LLaMA-Factory/data}/"
```

Regenerating these from the HuggingFace datasets instead is possible but **not
recommended** — the public FeTaQA / TabFact datasets have been revised since,
which shifts the path vocabulary by up to 3%. See
`tpe_impl/preprocess/compat/legacy_colrow_prompt_format.py` for the shim and
the measured drift if you need to do it anyway.

*HiTab and WikiTQ* — generate with the CLI driver `table_preprocess.py`, using
`--prompt_tuning True` for the `[TAB]/[ROW]/[CELL]` special-token format. This
step writes to a `data/` directory relative to the working directory rather
than to `$DEEPTABLE_DATA_DIR` — that is upstream TableLoRA's behaviour, left
unchanged so this stage stays directly comparable with the official release.

```bash
mkdir -p data
for ds in hitab wikitq; do
    python table_preprocess.py \
        --dataset_name "$ds" \
        --prompt_tuning True \
        --model_name deepseek-ai/deepseek-llm-7b-chat \
        --max_length 2000
    mv "data/${ds}_train.json" "data/${ds}_special_train.json"
    mv "data/${ds}_test.json"  "data/${ds}_special_test.json"
done
mv data/*_special_*.json "${DEEPTABLE_DATA_DIR:-LLaMA-Factory/data}/"
```

The rename to `*_special_*` prevents a later run with different flags from
overwriting them, and matches the names in `dataset_info.json`. Do not let
these overwrite the bare `wikitq_train.json` you copied from TableLoRA — the
two are different formats and step 2 reads both.

**2. TPE per-cell metadata + path vocabulary** (Section 3.4). Only HiTab uses
the plain per-cell metadata directly; WikiTQ, FeTaQA, and TabFact are flat
tables (no row-header hierarchy), so for those three, use the
`..._leftpath_variant.py` scripts' output — this is the data that produces the
paper's reported numbers for those benchmarks, not an optional ablation.

> **Filenames.** The `..._leftpath_variant.py` scripts write
> `{wikitq,fetaqa,tabfact}_tpe_v2_{train,test}.json` — the `_v2_` suffix is how
> the leftpath variant is named on disk and in `dataset_info.json`. So a WikiTQ
> run sets `dataset: wikitq_tpe_v2_train`, **not** `wikitq_tpe_train`. HiTab
> has no variant and stays `hitab_tpe_train`. This applies to every row of
> Table 2, not just the TPE ones. See the table in
> [`configs/README.md`](configs/README.md).
```bash
# Per-cell metadata (row path / column path) for each benchmark:
python tpe_impl/preprocess/build_hitab_tpe_dataset.py
python tpe_impl/preprocess/build_wikitq_tpe_dataset.py
python tpe_impl/preprocess/build_wikitq_tpe_dataset_leftpath_variant.py
python tpe_impl/preprocess/build_fetaqa_tabfact_tpe_dataset.py
python tpe_impl/preprocess/build_fetaqa_tpe_dataset_leftpath_variant.py
python tpe_impl/preprocess/build_tabfact_tpe_dataset_leftpath_variant.py

# Path-node vocabulary (Section 3.4), built incrementally. Run these in order —
# each one appends to the file the previous one produced:
python tpe_impl/preprocess/build_hitab_path_vocab.py
python tpe_impl/preprocess/build_combined_vocab.py
python tpe_impl/preprocess/extend_combined_vocab.py
python tpe_impl/preprocess/extend_combined_vocab_wikitq_leftpath_variant.py
python tpe_impl/preprocess/extend_combined_vocab_fetaqa_leftpath_variant.py
python tpe_impl/preprocess/extend_combined_vocab_tabfact_leftpath_variant.py
```
All of these read and write `$DEEPTABLE_DATA_DIR`. The result is
`combined_path_vocab.json` (all four benchmarks) alongside the HiTab-only
`hitab_tpe_path_vocab.json`; point a run's `path_vocab_path` at whichever
matches its `dataset`.

**3. Register the datasets and install the overlay.** `setup.sh` already did
this in step 0. To redo it by hand: merge
`llamafactory_src_overlay/dataset_info.json` into
`$DEEPTABLE_DATA_DIR/dataset_info.json`, and copy the eleven files under
`llamafactory_src_overlay/llamafactory/` over their counterparts in
`$DEEPTABLE_LF_ROOT/src/llamafactory/`.

**4. Train.** Four configurations per (backbone, dataset), matching Table 2.
Ready-to-run YAMLs for all four are in [`configs/`](configs/) (HiTab +
DeepSeek-7B; see `configs/README.md` for switching benchmark):

| Configuration | Flags | Launch |
|---|---|---|
| TableLoRA baseline | neither | `llamafactory-cli train configs/hitab_tablelora.yaml` |
| SAB only | `use_sab: true` | `SAB_LR_MULTIPLIER=1000 llamafactory-cli train configs/hitab_sab.yaml` |
| TPE only | `use_tpe: true` | `python -m tpe_impl.bin.run_tpe_training configs/hitab_tpe.yaml` |
| Full (SAB + TPE) | both | `SAB_LR_MULTIPLIER=1000 python -m tpe_impl.bin.run_tpe_training configs/hitab_sab_tpe.yaml` |

Four things are easy to get wrong here:

- **`use_sab: true` requires `emb_lora: true`** (SAB consumes the `row_ids` /
  `col_ids` that TableLoRA's data path emits). The flags are defined in
  `hparams/data_args.py`; TPE's keys are read separately by
  `tpe_impl/config/tpe_config.py` and stripped before LLaMA-Factory parses the
  YAML.
- **Any run with `use_tpe: true` must go through `run_tpe_training.py`.** That
  wrapper is what installs the collator / processor / `Linear_new` patches and
  registers the callback that writes `tpe_modules.safetensors`. Launching a TPE
  YAML with `llamafactory-cli` silently trains a plain TableLoRA baseline.
- **The learning-rate multipliers are set differently for the two modules.**
  SAB's is the `SAB_LR_MULTIPLIER` *environment variable* (read in
  `train/tuner.py`); TPE's is the `tpe_lr_multiplier` *YAML key*. Appendix G
  uses λ = 1000 for both, shared across all backbones and datasets. Both
  modules have very few parameters — 1,920 scalars for SAB on DeepSeek — so the
  base learning rate of 5e-6 barely moves them. Forgetting `SAB_LR_MULTIPLIER`
  runs SAB at 1× and will not reproduce the paper's numbers.
- **All four rows use the same dataset.** For a given benchmark, the baseline
  and SAB-only rows read the same `*_tpe_*` file as the TPE rows;
  LLaMA-Factory's aligner drops the extra metadata column when `use_tpe` is
  off. There is no separate baseline dataset to build.

**5. Predict**, using the same command as training but with `do_predict: true`
(and `do_train: false`) in the YAML, and `adapter_name_or_path` pointed at the
trained adapter checkpoint. TPE runs must still be launched through
`run_tpe_training.py`, which is what reloads `tpe_modules.safetensors` from the
adapter directory. This produces `generated_predictions.jsonl` in the output
directory.

**6. Score** the predictions:
```bash
python evaluation/eval_accuracy.py    <predict_dir>   # HiTab (exact-match, HiTab's own hmt_score)
python evaluation/eval_wikitq.py      <predict_dir>   # WikiTQ (official set-comparison semantics)
python evaluation/eval_fetaqa_bleu.py <predict_dir>   # FeTaQA (corpus BLEU-4 via sacrebleu)
python evaluation/eval_tabfact.py     <predict_dir>   # TabFact (accuracy + True/False breakdown)
```
Each script prints the metric reported in Table 2 (and, for `eval_tabfact.py`,
the True/False breakdown used in Appendix F). Passing several directories at
once prints a comparison summary.

**7. (Optional) Sanity checks.** The scripts in `tpe_impl/tests/` are run
directly, e.g. `python tpe_impl/tests/test_tpe_path_module.py`. Any that need
data, a GPU, or a trained adapter you do not have print `SKIP: ...` and exit 0
rather than failing. `test_tpe_path_module.py` needs nothing at all; the
`test_bit_exact_equivalence_*.py` scripts implement the zero-initialization
verification described in Appendix H, and the full-model one needs
`DEEPTABLE_TEST_ADAPTER` pointed at an adapter from step 4.

---

## 1. `llamafactory_src_overlay/` — SAB and its integration into LLaMA-Factory

Overlay patches on top of LLaMA-Factory v0.9.1 + PEFT 0.12.0. Each file below
replaces the corresponding upstream LLaMA-Factory file.

| File | Ours / vendored | Purpose |
|---|---|---|
| `llamafactory/sab/sab.py` | **Ours** | **Structural Attention Bias (Section 3.3).** Monkey-patches `LlamaAttention` / `LlamaSdpaAttention` / `Qwen2Attention` / `Qwen2SdpaAttention` (eager and SDPA) to add the learnable per-layer, per-head row/column structural bias to the attention logits (Eq. 2). Also computes the same-row / same-column indicator matrices from `row_ids`/`col_ids`, and handles save/load of the `alpha_row`/`alpha_col` parameters. |
| `llamafactory/sab/__init__.py` | Ours | Package exports for `sab.py`. |
| `llamafactory/data/processors/supervised.py` | Patched (LlamaFactory) | Supervised-example preprocessor, patched to additionally emit `row_ids`/`col_ids` per token (the flat row/column coordinates used by both TableLoRA's 2D-LoRA and by SAB's same-row/same-column masks). |
| `llamafactory/data/processors/unsupervised.py` | Patched (LlamaFactory) | Same `row_ids`/`col_ids` patch as above, for the unsupervised (inference-time) data path. |
| `llamafactory/data/template.py` | Vendored (LlamaFactory) | Chat-template handling; included as a dependency of the two processors above. No SAB/TPE-specific logic. |
| `llamafactory/hparams/data_args.py` | Patched (LlamaFactory) | Adds the `use_sab` and `emb_lora` CLI/YAML flags that turn on SAB and the row/column coordinate embeddings respectively. |
| `llamafactory/model/adapter.py` | Patched (LlamaFactory) | Model-loading path; adds the `TABLE_LORA_ENABLED` environment-variable hook used to attach the TableLoRA 2D-LoRA adapter. |
| `llamafactory/train/tuner.py` | Patched (LlamaFactory) | Training entry point; wires up TableLoRA / SAB installation based on the CLI flags before training starts. |
| `llamafactory/table_lora/table_lora.py` | Vendored (TableLoRA / Microsoft) | The TableLoRA 2D-LoRA baseline (Section 3.1) that SAB extends. Not our contribution; included so `sab.py` and the processors above are readable in context. |
| `llamafactory/table_lora/prompt_tuning.py` | Vendored (TableLoRA / Microsoft) | `[TAB]`/`[ROW]`/`[CELL]` special-token definitions and prompt construction used by TableLoRA and by `table_preprocess/`. |
| `llamafactory/table_lora/__init__.py` | Vendored (TableLoRA / Microsoft) | Package exports. |
| `dataset_info.json` | Ours (config, not code) | LLaMA-Factory dataset registry entries for the preprocessed HiTab/WikiTQ/FeTaQA/TabFact files. |

## 2. `tpe_impl/` — Tree Path Encoding (Section 3.4)

Our implementation of TPE, kept as a separate package from the LLaMA-Factory
overlay so it can be installed/removed independently of SAB (matching the
SAB-only / TPE-only / combined ablations in Table 2).

### `tpe_impl/module/` — the TPE module itself
| File | Purpose |
|---|---|
| `tpe_path_module.py` | `TPEPathModule` / `TPEPathModuleList`: per-layer path-embedding tables (`Emb_prow`, `Emb_pcol` in the paper) and the mean/sum-pooling operation over row-/column-header-tree path nodes. |
| `monkey_patch.py` | Patches TableLoRA's `Linear_new.forward` to add the pooled TPE path embeddings into the same `B_tab` projection used by the 2D-LoRA coordinate embeddings (the combined equation in Section 3.4). Zero-initialized so an untrained TPE module is a bit-exact no-op. |
| `save_load.py` | Saves/loads the TPE module's state (`tpe_modules.safetensors`, `tpe_path_vocab.json`, `tpe_metadata.json`), since PEFT's own adapter save/load only handles `lora_*`-prefixed parameters. |

### `tpe_impl/collator/` and `tpe_impl/processors/` — data pipeline (Appendix H, "Metadata pipeline")
| File | Purpose |
|---|---|
| `tpe_collator.py` | Patches `DataCollatorForSeq2Seq` to pad the per-token row-/column-header-path-ID sequences to a common depth within each batch (3D padding, using `PATH_PAD = -1`). |
| `tpe_supervised.py` | TPE-aware version of the supervised preprocessor: looks up each token's per-cell metadata (row/column header path) and emits the path-ID sequences alongside the standard `input_ids`/`labels`. |
| `metadata_index.py` | Builds a prompt-hash → per-cell-metadata lookup index once per run, since LLaMA-Factory's own data aligner drops the extra metadata column before it reaches the processor. |

### `tpe_impl/config/`
| File | Purpose |
|---|---|
| `tpe_config.py` | `TPEConfig` dataclass: reads the `use_tpe` master flag, path-vocab location, embedding rank, learning-rate multiplier, pooling mode, and ablation switches (`disable_2d_lora`, `tpe_top_only`) from a training YAML or environment variables. |

### `tpe_impl/bin/`
| File | Purpose |
|---|---|
| `run_tpe_training.py` | Standalone entry point that installs the TPE patches (collator, processor swap, attention patch, optimizer param-group split for the learning-rate multiplier) around a call into LLaMA-Factory's own training loop. |

### `tpe_impl/preprocess/` — building the per-cell metadata and path vocabulary for each dataset
| File | Purpose |
|---|---|
| `render_from_hmt.py` | Renders a HiTab table (from its raw header-tree JSON) directly into the canonical `[TAB][ROW][CELL]` token sequence plus per-cell metadata (row path, column path). Used by `build_hitab_tpe_dataset.py`. |
| `build_hitab_tpe_dataset.py` | Builds `hitab_tpe_{train,test}.json`: combines each raw HiTab question/answer with the rendered prompt and per-cell metadata. |
| `build_wikitq_tpe_dataset.py` | WikiTQ, intermediate step. WikiTQ has only flat (single-row) column headers and no row-header hierarchy, so `left_path` is always empty here; `top_path` is the single column header. **Not what training actually reads** — see `..._leftpath_variant.py` below. |
| `build_fetaqa_tabfact_tpe_dataset.py` | Same idea for FeTaQA and TabFact, intermediate step; parses the `/* ... */` table block already present in the TableLoRA-style prompt and re-serializes it into the canonical format. Requires the `<COL>`/`<ROW>` input format produced via the compat shim — see "How to reproduce" step 1. Also not what training actually reads. |
| `build_wikitq_tpe_dataset_leftpath_variant.py`, `build_fetaqa_tpe_dataset_leftpath_variant.py`, `build_tabfact_tpe_dataset_leftpath_variant.py` | **This is the data actually used to produce the paper's WikiTQ/FeTaQA/TabFact numbers** (confirmed against the real training configs' `dataset:` field), not an optional side-ablation. Since these three benchmarks are flat tables with no natural row-header hierarchy, `left_path` in the intermediate file above is always empty; these scripts synthesize a non-empty `left_path = [first_col_header, first_col_value]` from the table's first column instead. |
| `build_hitab_path_vocab.py` | Builds the path-node vocabulary (`<PAD_PATH>`, `<UNK_PATH>`, then every distinct header-node **value string** seen in the HiTab training data) that `tpe_path_module.py` uses for embedding lookups — this is the vocabulary construction referenced in Section 3.4. |
| `build_combined_vocab.py` | Merges the HiTab and WikiTQ path vocabularies into one shared vocabulary (preserving HiTab's existing IDs). |
| `extend_combined_vocab.py` | Extends the combined vocabulary with FeTaQA and TabFact column-header values. |
| `extend_combined_vocab_wikitq_leftpath_variant.py`, `extend_combined_vocab_fetaqa_leftpath_variant.py`, `extend_combined_vocab_tabfact_leftpath_variant.py` | Also required, not optional: adds the new `left_path` leaf-value vocabulary entries introduced by the `..._leftpath_variant.py` datasets above, appending in place to the same combined vocabulary file. |
| `inspect_hitab_rendering_samples.py` | Diagnostic script: dumps side-by-side renders of a handful of HiTab samples (old TableInstruct format vs. our canonical render) for manual inspection during development. |
| `analyze_oov_depth.py` | Diagnostic script: reports the out-of-vocabulary rate of test-set path nodes, broken down by depth in the header tree. |
| `compat/legacy_colrow_prompt_format.py` | **Added by us** (not part of the original pipeline): a reconstruction of the pre-rewrite `<COL>`/`<ROW>` table-prompt format that `build_fetaqa_tabfact_tpe_dataset.py` needs as input. See "How to reproduce" step 1. |

### `tpe_impl/tests/` — unit / smoke tests

Run directly (`python tpe_impl/tests/<name>.py`). Each skips with a `SKIP:`
line and exit code 0 when a prerequisite (preprocessed data, a GPU, a trained
adapter) is missing, so none of them hard-fail on a fresh checkout.

| File | Purpose |
|---|---|
| `_bootstrap.py` | Shared `sys.path` setup plus the `require_data` / `require_vocab` / `require_gpu` / `require_adapter` skip helpers. Not a test. |
| `test_tpe_path_module.py` | Unit tests for `TPEPathModule.pool()` (empty / partial / full / all-padding path inputs). |
| `test_tpe_collator.py` | Verifies the collator produces correctly shaped, padded batches. |
| `test_tpe_supervised_processor.py` | Smoke test: runs the TPE-aware processor on one real HiTab sample and checks token-level path alignment. |
| `test_padding.py` | End-to-end check that padding uses `PATH_PAD` correctly and that all-padding tokens pool to a zero vector (not `NaN`). |
| `test_tpe_save_load.py` | Save/load round-trip test for the TPE module state. |
| `test_vocab_coverage.py` | Checks the path vocabulary is internally consistent with the training data it was built from. |
| `test_bit_exact_equivalence_fullmodel.py` | **Corresponds to the "Zero-initialization verification" described in Appendix H:** confirms that a full 7B model with the TPE (and SAB) patches installed but untrained produces bit-identical logits to the unpatched TableLoRA baseline. Needs a GPU and `DEEPTABLE_TEST_ADAPTER` set to an adapter from step 4. |
| `test_bit_exact_equivalence_micro.py` | Same check, on a single `Linear` layer instead of the full model (fast, for iterating on the patch logic). |
| `test_bit_exact_equivalence_minimal_real_model.py` | Same check again, minimal single-forward-pass version against a real (but freshly-initialized) model. |

## 3. `table_preprocess.py` and `table_preprocess/` — table serialization (Section 3.2)

**Vendored from the official TableLoRA release** (Copyright Microsoft
Corporation, MIT License), not our contribution. Builds the initial
`[TAB][ROW][CELL]`-tokenized prompts (one file per benchmark) that
`tpe_impl/preprocess/` then augments with per-cell TPE metadata.

| File | Purpose |
|---|---|
| `table_preprocess.py` (top level) | **CLI entry point** (`--dataset_name`, `--model_name`, `--max_length`, `--prompt_tuning`); dispatches to the function of the matching name in `table_preprocess/`. This is what you actually run — see "How to reproduce" step 1. |
| `table_preprocess/hitab.py`, `wikitq.py`, `fetaqa.py`, `tabfact.py` | Per-dataset loading + serialization into the `[TAB]/[ROW]/[CELL]` prompt format, using TableLoRA's own `prompt_tuning_table_prompt`. Each writes `data/{dataset}_{train,test}.json`, which must be renamed to `..._special_{train,test}.json` before use (see step 1) to avoid being overwritten by a subsequent run with different flags. `hitab.py` carries one local change: the raw HiTab file locations come from `deeptable_paths.py` and are loaded lazily, instead of the upstream hard-coded `src/table_preprocess/hitab/`. |
| `table_preprocess/hitab_table_type.py` | Helper: computes header-tree depth from HiTab's raw header-tree JSON. |

## 4. `evaluation/` — scoring scripts

Take a `predict_dir` (containing `generated_predictions.jsonl`) and print the
metric reported in the paper. These are the exact scripts used to produce every
number in Table 2, 4, 5, 6, 8, 9, 10, and 11.

| File | Purpose |
|---|---|
| `eval_accuracy.py` | HiTab: exact-match accuracy using HiTab's own official `hmt_score` normalization (number parsing, `%`/`$`/comma handling, string normalization). |
| `eval_wikitq.py` | WikiTQ: the official WikiTableQuestions comparison semantics (gold answers as an unordered, number-aware set), which differs from HiTab's ordered-list comparison. |
| `eval_fetaqa_bleu.py` | FeTaQA: corpus BLEU-4 via `sacrebleu`, matching the FeTaQA paper's own evaluation protocol. |
| `eval_tabfact.py` | TabFact: accuracy plus a breakdown by gold label (True/False), used for Appendix F. |

## 5. `configs/` — training configurations

| File | Purpose |
|---|---|
| `hitab_tablelora.yaml`, `hitab_sab.yaml`, `hitab_tpe.yaml`, `hitab_sab_tpe.yaml` | The four rows of Table 2, on HiTab + DeepSeek-7B. Structurally complete and runnable; values swept in Appendix G are marked `# Appendix G`. |
| `configs/README.md` | Which launcher each config needs, and the `dataset` / `cutoff_len` / `path_vocab_path` substitutions for the other three benchmarks. |

## 6. Top level

| File | Purpose |
|---|---|
| `deeptable_paths.py` | Single source of truth for every filesystem location, read from `DEEPTABLE_*` environment variables with repo-relative defaults. Imported by `tpe_impl/`, `table_preprocess/hitab.py`, and the tests. |
| `requirements.txt` | Pinned dependency versions (PyTorch 2.5.1, Transformers 4.46.1, PEFT 0.12.0, etc.) used for every reported experiment. |
| `setup.sh` | Environment setup: clones LLaMA-Factory v0.9.1, applies the overlay in `llamafactory_src_overlay/`, merges `dataset_info.json`, installs the pinned dependencies, and downloads the raw HiTab files. |
| `LICENSE`, `NOTICE` | MIT for our code; `NOTICE` gives the per-directory breakdown and reproduces the TableLoRA (MIT) and LLaMA-Factory (Apache-2.0) licenses. |
| `CITATION.cff` | Citation metadata. |
