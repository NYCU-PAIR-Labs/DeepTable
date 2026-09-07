"""Structural Attention Bias (SAB) — non-invasive add-on to TableLoRA.

Public surface:
    - SABModule           — holds the learnable per-(layer, head) row/col scalars
    - load_sab(model)     — install the SAB hook + monkey-patch on a loaded model
    - unload_sab(model)   — undo install_sab (rare; mainly for tests)
    - set_sab_inputs(...) — manually push row_ids/col_ids (used by tests)
    - clear_sab_inputs()  — clear cached masks
"""

from .sab import (
    SABModule,
    load_sab,
    unload_sab,
    install_sab_attention_patch,
    uninstall_sab_attention_patch,
    set_sab_inputs,
    clear_sab_inputs,
    save_sab,
    load_sab_state,
    SAB_STATE_FILENAME,
    _SAB_CTX,
)

__all__ = [
    "SABModule",
    "load_sab",
    "unload_sab",
    "install_sab_attention_patch",
    "uninstall_sab_attention_patch",
    "set_sab_inputs",
    "clear_sab_inputs",
    "save_sab",
    "load_sab_state",
    "SAB_STATE_FILENAME",
    "_SAB_CTX",
]
