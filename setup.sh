#!/usr/bin/env bash
# DeepTable — environment setup
#
# This script:
#   1. Clones LLaMA-Factory at the exact tag we depend on (v0.9.1)
#   2. Overlays our modified source files on top of it (SAB + row/col-id plumbing)
#   3. Installs LLaMA-Factory in editable mode
#   4. Installs the pinned Python dependencies
#   5. Downloads the raw HiTab files that table_preprocess/hitab.py needs
#
# Usage (from the repo root):
#   bash setup.sh
#
# Everything is placed relative to this repo, matching the defaults in
# deeptable_paths.py. To put LLaMA-Factory elsewhere, set DEEPTABLE_LF_ROOT
# before running this script AND keep it set for every later command:
#   export DEEPTABLE_LF_ROOT=/path/to/LLaMA-Factory
#
# After this finishes, follow README.md "How to reproduce" from step 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_DIR="${DEEPTABLE_LF_ROOT:-${REPO_ROOT}/LLaMA-Factory}"
HITAB_DIR="${DEEPTABLE_HITAB_DIR:-${REPO_ROOT}/table_preprocess/hitab}"
LF_TAG="v0.9.1"

echo "============================================"
echo "[1/5] Cloning LLaMA-Factory ${LF_TAG}"
echo "  -> ${LF_DIR}"
echo "============================================"
if [ -d "${LF_DIR}/.git" ]; then
    echo "  LLaMA-Factory already cloned, skipping git clone."
else
    git clone --depth 1 --branch "${LF_TAG}" \
        https://github.com/hiyouga/LLaMA-Factory.git "${LF_DIR}"
fi

echo
echo "============================================"
echo "[2/5] Applying our overlay onto LLaMA-Factory"
echo "============================================"
# Copy our modified source files into the LLaMA-Factory source tree.
# These are the only files we change; everything else stays vanilla v0.9.1.
OVERLAY="${REPO_ROOT}/llamafactory_src_overlay/llamafactory"
TARGET="${LF_DIR}/src/llamafactory"

if [ ! -d "${TARGET}" ]; then
    echo "ERROR: ${TARGET} does not exist — is ${LF_DIR} a LLaMA-Factory checkout?" >&2
    exit 1
fi

mkdir -p "${TARGET}/table_lora" "${TARGET}/sab"

for rel in \
    data/processors/supervised.py \
    data/processors/unsupervised.py \
    data/template.py \
    hparams/data_args.py \
    model/adapter.py \
    train/tuner.py \
    table_lora/__init__.py \
    table_lora/table_lora.py \
    table_lora/prompt_tuning.py \
    sab/__init__.py \
    sab/sab.py
do
    cp "${OVERLAY}/${rel}" "${TARGET}/${rel}"
    echo "  ${rel}"
done

# Dataset registry. Merge rather than overwrite if the user already has entries.
mkdir -p "${LF_DIR}/data"
DATASET_INFO="${LF_DIR}/data/dataset_info.json"
if [ -f "${DATASET_INFO}" ]; then
    python - "${DATASET_INFO}" "${REPO_ROOT}/llamafactory_src_overlay/dataset_info.json" <<'PYEOF'
import json, sys
target_path, ours_path = sys.argv[1], sys.argv[2]
with open(target_path) as f:
    target = json.load(f)
with open(ours_path) as f:
    ours = json.load(f)
added = [k for k in ours if k not in target]
target.update(ours)
with open(target_path, "w") as f:
    json.dump(target, f, indent=2)
print(f"  dataset_info.json: merged {len(ours)} entries ({len(added)} new)")
PYEOF
else
    cp "${REPO_ROOT}/llamafactory_src_overlay/dataset_info.json" "${DATASET_INFO}"
    echo "  dataset_info.json: installed"
fi

echo "  Overlay applied."

echo
echo "============================================"
echo "[3/5] Installing LLaMA-Factory (editable)"
echo "============================================"
pip install -e "${LF_DIR}[torch,metrics]"

echo
echo "============================================"
echo "[4/5] Installing pinned Python deps"
echo "============================================"
# These pins override anything LLaMA-Factory's setup.py installs, so they must
# run AFTER `pip install -e .` above.
pip install -r "${REPO_ROOT}/requirements.txt"

echo
echo "============================================"
echo "[5/5] Downloading raw HiTab files"
echo "  -> ${HITAB_DIR}"
echo "============================================"
# table_preprocess/hitab.py and tpe_impl/preprocess/build_hitab_tpe_dataset.py
# read three things from this directory:
#   train_samples.jsonl      (from microsoft/HiTab)
#   hmt/{table_id}.json      (from microsoft/HiTab)
#   hitab_test.json          (from osunlp/TableInstruct)
# They are not bundled in this repo (size + licensing).
mkdir -p "${HITAB_DIR}/hmt"

if [ ! -f "${HITAB_DIR}/train_samples.jsonl" ]; then
    echo "  Cloning microsoft/HiTab to a temp dir to grab the raw files..."
    TMP_HITAB="$(mktemp -d)"
    trap 'rm -rf "${TMP_HITAB}"' EXIT
    git clone --depth 1 https://github.com/microsoft/HiTab.git "${TMP_HITAB}"
    cp "${TMP_HITAB}/data/train_samples.jsonl" "${HITAB_DIR}/train_samples.jsonl"
    # The hierarchical-matrix-table JSONs; HiTab has moved these between
    # releases, so try the known locations in order.
    if ls "${TMP_HITAB}"/data/tables/raw/*.json >/dev/null 2>&1; then
        cp "${TMP_HITAB}"/data/tables/raw/*.json "${HITAB_DIR}/hmt/"
    elif ls "${TMP_HITAB}"/data/raw/*.json >/dev/null 2>&1; then
        cp "${TMP_HITAB}"/data/raw/*.json "${HITAB_DIR}/hmt/"
    else
        find "${TMP_HITAB}" -name "*.json" -path "*hmt*" -exec cp {} "${HITAB_DIR}/hmt/" \;
    fi
    N_HMT=$(ls -1 "${HITAB_DIR}/hmt/" | wc -l)
    if [ "${N_HMT}" -eq 0 ]; then
        echo "ERROR: found no hmt table JSONs in the HiTab clone. Copy them to" >&2
        echo "       ${HITAB_DIR}/hmt/ manually." >&2
        exit 1
    fi
    echo "  Copied ${N_HMT} hmt table files."
else
    echo "  HiTab raw files already present, skipping."
fi

if [ ! -f "${HITAB_DIR}/hitab_test.json" ]; then
    echo "  Downloading hitab_test.json from osunlp/TableInstruct..."
    HITAB_DIR="${HITAB_DIR}" python - <<'PYEOF'
import json
import os

from datasets import load_dataset

hitab_dir = os.environ["HITAB_DIR"]
ds = load_dataset(
    "osunlp/TableInstruct",
    data_files="eval_data/in_domain_test/hitab_test.json",
    split="train",
    trust_remote_code=True,
)
out = [dict(s) for s in ds]
with open(os.path.join(hitab_dir, "hitab_test.json"), "w") as f:
    json.dump(out, f)
print(f"  Wrote {len(out)} test samples.")
PYEOF
else
    echo "  hitab_test.json already present, skipping."
fi

echo
echo "============================================"
echo "Setup complete."
echo "============================================"
echo
echo "  LLaMA-Factory : ${LF_DIR}"
echo "  Data dir      : ${LF_DIR}/data"
echo "  HiTab raw     : ${HITAB_DIR}"
echo
echo "Next: README.md 'How to reproduce', step 1 (table serialization). E.g."
echo "  mkdir -p data"
echo "  python table_preprocess.py --dataset_name hitab --prompt_tuning True \\"
echo "      --model_name deepseek-ai/deepseek-llm-7b-chat --max_length 2000"
