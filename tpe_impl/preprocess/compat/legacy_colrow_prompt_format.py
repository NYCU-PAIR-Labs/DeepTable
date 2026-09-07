"""Reconstruction of the `<COL>`/`<ROW>` table-prompt format that
`build_fetaqa_tabfact_tpe_dataset.py` expects as input for FeTaQA/TabFact.

**You probably do not need this file.** The FeTaQA/TabFact inputs used for the
paper's reported runs are the data files that ship with the official TableLoRA
release — `data/{fetaqa,tabfact}_{train,test}.json` in
https://github.com/microsoft/TableLoRA — copied verbatim (verified by md5).
They are already in the `<COL>`/`<ROW>` format, so the normal path is simply to
download them and point `DEEPTABLE_RAW_TABLELORA_DIR` at that directory. See
README step 1.

This module exists only for the case where you want to regenerate those inputs
from the HuggingFace datasets yourself rather than using TableLoRA's frozen
copies. `table_lora/prompt_tuning.py` emits `[TAB]/[ROW]/[CELL]` tokens, so
regenerating requires shadowing it with the older `<TABLE>/<ROW>/<COL>` scheme
that TableLoRA's own `table_lora.py` still declares (unused) in its
`TABLE_TOKEN` enum. This file is that shim.

Be aware that regenerating does **not** reproduce the paper's inputs exactly:
the public `DongfuJiang/FeTaQA` / `tab_fact` HuggingFace datasets have been
revised since TableLoRA froze its copies. Rebuilding the whole vocabulary chain
from regenerated inputs gives:

    stage                          regenerated   paper (TableLoRA's files)
    HiTab only                     16,793        16,793   (exact)
    + WikiTQ (v1)                  19,767        19,767   (exact)
    + FeTaQA/TabFact (v1)          33,693        33,882   (-0.6%)
    + WikiTQ leftpath (v2)         43,240        43,373   (-0.3%)
    + FeTaQA leftpath (v2)         64,148        63,246   (+1.4%)
    + TabFact leftpath (v2)       109,580       106,524   (+2.9%)

HiTab and WikiTQ reproduce exactly because they do not depend on the revised
datasets. The drift is dataset-revision drift, not a structural error in this
shim — the test-set sizes matched exactly (FeTaQA 2,003 / TabFact 12,779, both
matching Table 1). Use TableLoRA's shipped files to avoid it entirely.

Usage (only if regenerating): shadow the real
`llamafactory.table_lora.prompt_tuning` module with this one by putting a
package named `llamafactory` on PYTHONPATH ahead of your real LLaMA-Factory
install, containing `table_lora/prompt_tuning.py` = this file. Do this ONLY for
`table_preprocess.py --dataset_name fetaqa` / `--dataset_name tabfact`; HiTab
and WikiTQ must keep the real, `[TAB]/[ROW]/[CELL]`-emitting version.
"""
from enum import Enum
import pandas as pd


class TABLE_TOKEN(Enum):
    TABLE = "<TABLE>"
    ROW = "<ROW>"
    COL = "<COL>"


def prompt_tuning_table_prompt(table: pd.DataFrame) -> str:
    parts = [TABLE_TOKEN.COL.value.join(str(c) for c in table.columns)]
    for _, row in table.iterrows():
        parts.append(TABLE_TOKEN.ROW.value + TABLE_TOKEN.COL.value.join(str(v) for v in row.values))
    return "".join(parts)
