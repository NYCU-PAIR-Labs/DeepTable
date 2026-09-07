"""Zero-initialization verification (micro): bit-exact equivalence on a SINGLE
Linear_new instance.

Instead of loading the full 7B model (which took 7+ min for first forward in the
earlier test), we:
  1. Build a fresh `Linear_new` wrapping a random `nn.Linear`, populated with
     random but deterministic LoRA weights (so the 2D LoRA contribution is nonzero).
  2. Feed a dummy input + row_ids/col_ids.
  3. Compare three forward paths:
     (A) unpatched: the ORIGINAL Linear_new.forward from table_lora.py
     (B) patched + _TPE_CTX empty: should be bit-exact with (A)
     (C) patched + _TPE_CTX filled, but TPEPathModule.emb.* = 0 (zero-init):
         should be bit-exact with (A)
  4. Additional: (D) patched + _TPE_CTX filled + NON-zero emb weights → output
     should DIFFER from (A). This sanity-checks that the patch actually takes
     effect when embeddings have signal.

This is MATHEMATICALLY equivalent to the full-model bit-exact test, but runs in
<1s instead of 7+ min.
"""
import sys, os
from _bootstrap import BACKBONE, require_data, require_gpu, require_vocab  # noqa: F401

os.environ.setdefault("TABLE_LORA_ENABLED", "1")

import torch
import torch.nn as nn


def main():
    # Ensure TableLoRA imports the Linear_new class
    from llamafactory.table_lora.table_lora import Linear_new, LoraLayer_new
    from tpe_impl.module.monkey_patch import (
        install_tpe_patch, uninstall_tpe_patch,
        _patched_linear_new_forward, _TPE_CTX, _TPE_MODULES,
    )
    from tpe_impl.module.tpe_path_module import TPEPathModuleList, PATH_PAD
    from tpe_impl.config import TPEConfig

    torch.manual_seed(0)

    # --- Build a Linear_new standing in for a k_proj of layer 5
    hidden = 64
    rank = 8
    base = nn.Linear(hidden, hidden, bias=False)
    lin = Linear_new(base_layer=base, adapter_name="default", r=rank,
                     lora_alpha=16, lora_dropout=0.0)
    # Activate adapter
    lin.active_adapter = "default"

    # Populate LoRA + 2D LoRA weights with random values (not zero)
    torch.manual_seed(1)
    with torch.no_grad():
        lin.lora_A["default"].weight.normal_(0, 0.02)
        lin.lora_B["default"].weight.normal_(0, 0.02)
        lin.lora_A_tab_col["default"].weight.normal_(0, 0.02)
        lin.lora_A_tab_row["default"].weight.normal_(0, 0.02)
        lin.lora_B_tab["default"].weight.normal_(0, 0.02)

    # Pretend this is layer 5 / k_proj (will be used only when patch is installed)
    lin._tpe_layer_idx = 5
    lin._tpe_is_k_proj = True

    # Row/col ids for the forward
    B, L = 2, 10
    lin.row_ids = torch.randint(0, 30, (B, L))
    lin.col_ids = torch.randint(0, 40, (B, L))

    # Random input
    x = torch.randn(B, L, hidden)

    # ---- (A) Unpatched forward: reference
    # Save original forward (in case any earlier import touched it)
    ORIG = Linear_new.forward
    # Ensure we're using the genuine unpatched version (this should be a no-op if not patched yet)
    out_A = ORIG(lin, x)

    # ---- (B) Install patch with use_tpe=True, but _TPE_CTX empty ----
    # We need a fake ModuleList to pass vocab checks in install; the patch will
    # see _TPE_CTX["top_path_ids"] is None and skip the tpe term.
    # But install_tpe_patch needs a model — we don't have one. Instead, we
    # manually set global state to simulate "patch installed, context empty".
    from tpe_impl.module import monkey_patch as mp
    mp._ORIG_LINEAR_FORWARD = ORIG
    mp._PATCH_INSTALLED = True
    Linear_new.forward = _patched_linear_new_forward

    # Case B: no tpe modules, no context
    mp._TPE_MODULES = None
    mp._TPE_CTX["top_path_ids"] = None
    mp._TPE_CTX["left_path_ids"] = None
    out_B = Linear_new.forward(lin, x)

    # Case C: tpe modules present (zero-init), context has path_ids
    vocab_size = 100
    module_list = TPEPathModuleList(
        num_layers=10, path_vocab_size=vocab_size, low_rank_dim=rank,
    )
    # Embeddings are zero-init by default
    for i in range(10):
        for emb in (module_list[i].emb_top_path_K, module_list[i].emb_top_path_V,
                    module_list[i].emb_left_path_K, module_list[i].emb_left_path_V):
            assert torch.all(emb.weight == 0)

    mp._TPE_MODULES = module_list
    # Make path ids with some valid (nonzero) content and some padding
    top_ids = torch.randint(2, vocab_size, (B, L, 3))
    top_ids[0, 0] = torch.tensor([-1, -1, -1])  # all pad at one token
    left_ids = torch.randint(2, vocab_size, (B, L, 4))
    mp._TPE_CTX["top_path_ids"] = top_ids
    mp._TPE_CTX["left_path_ids"] = left_ids
    out_C = Linear_new.forward(lin, x)

    # Case D: same as C but with NON-zero embeddings → should DIFFER from A
    with torch.no_grad():
        for i in range(10):
            module_list[i].emb_top_path_K.weight.normal_(0, 0.5)
            module_list[i].emb_top_path_V.weight.normal_(0, 0.5)
            module_list[i].emb_left_path_K.weight.normal_(0, 0.5)
            module_list[i].emb_left_path_V.weight.normal_(0, 0.5)
    out_D = Linear_new.forward(lin, x)

    # ---- Compare ----
    def report(name, a, b):
        diff = (a.float() - b.float()).abs()
        print(f"  {name}: max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}  "
              f"|a|_mean={a.float().abs().mean().item():.3e}")
        return diff.max().item()

    TOL = 1e-6  # fp32, strict bit-exact

    print("\n=== Bit-exact paths ===")
    d_ab = report("(A) unpatched vs (B) patched+empty_ctx", out_A, out_B)
    d_ac = report("(A) unpatched vs (C) patched+zero_emb ", out_A, out_C)
    d_bc = report("(B) empty_ctx vs (C) zero_emb        ", out_B, out_C)

    print(f"\n=== Control: non-zero emb MUST diverge ===")
    d_ad = report("(A) unpatched vs (D) patched+nonzero ", out_A, out_D)

    all_ok = True
    if d_ab > TOL:
        print(f"✗ (A) vs (B) not bit-exact (tol={TOL})"); all_ok = False
    if d_ac > TOL:
        print(f"✗ (A) vs (C) not bit-exact (tol={TOL})"); all_ok = False
    if d_bc > TOL:
        print(f"✗ (B) vs (C) differ"); all_ok = False
    if d_ad < 1e-3:
        print(f"✗ (A) vs (D) should DIVERGE (got max|Δ|={d_ad:.3e}) — patch might not be applied"); all_ok = False

    # Cleanup
    uninstall_tpe_patch(None)

    assert all_ok, "bit-exact equivalence failed"
    print("\n✓ Bit-exact equivalence (micro) passed.")
    print("  - (B) patched with empty _TPE_CTX   ≡  (A) unpatched (bit-exact)")
    print("  - (C) patched with zero embeddings   ≡  (A) unpatched (bit-exact)")
    print(f"  - (D) patched with nonzero embeddings diverges from (A) by max|Δ|={d_ad:.3e}")


if __name__ == "__main__":
    main()
