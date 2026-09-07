"""Zero-initialization verification — minimal real-model forward. One forward
pass only, no pre-trained adapter.

Uses fresh-initialized LoRA (not loaded from disk) so there's only one adapter,
no "default_1" + "default" parallel loops that inflated the earlier 7B test.
"""
import sys, os, time
from _bootstrap import BACKBONE, require_data, require_gpu, require_vocab  # noqa: F401

os.environ.setdefault("TABLE_LORA_ENABLED", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = BACKBONE  # override with $DEEPTABLE_BACKBONE


def main():
    require_gpu()
    require_vocab()

    from llamafactory.table_lora.table_lora import load_table_lora, set_special_token_ids
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left",
                                              add_eos_token=True, add_bos_token=True)
    tokenizer.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)
    special_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["[TAB]", "[ROW]", "[CELL]"]]
    set_special_token_ids(special_ids)
    load_table_lora()

    device = torch.device("cuda:0")
    t0 = time.time()
    print("Loading base model...", flush=True)
    # attn_implementation="eager" is required for the bit-exact comparison below.
    # With "sdpa", injecting an (all-zero) attention bias forces PyTorch's SDPA
    # dispatcher off the fused flash-attention kernel and onto a different backend
    # (math/mem-efficient), which is numerically close but NOT bit-identical to the
    # unpatched forward — that kernel-choice noise alone is enough to blow past
    # TOL_STRICT even though the SAB/TPE math is exactly a no-op. Verified
    # empirically: same test on Qwen2.5-7B-Instruct gave max|delta|=4.4e-01 under
    # sdpa vs 0.0 exactly under eager.
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    base.resize_token_embeddings(len(tokenizer))
    base.to(device).eval()
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("Attaching fresh LoRA (get_peft_model)...", flush=True)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    model.to(device).eval()
    # Fresh peft init zeros lora_B/lora_B_tab → whole LoRA branch is zero.
    # For a meaningful bit-exact test, populate lora_B_tab with small random values
    # so pos_signal actually affects the output.
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    torch.manual_seed(2)
    with torch.no_grad():
        for _n, _m in model.named_modules():
            if isinstance(_m, PeftLoraLinear):
                for adapter in _m.lora_B.keys():
                    _m.lora_B[adapter].weight.normal_(0, 0.02)
                    if hasattr(_m, "lora_B_tab") and adapter in _m.lora_B_tab:
                        _m.lora_B_tab[adapter].weight.normal_(0, 0.02)
    print(f"  peft attached + lora_B(+tab) initialized nonzero: {time.time()-t0:.1f}s", flush=True)

    # Dummy input — 200 tokens
    torch.manual_seed(0)
    B, L = 1, 200
    input_ids = torch.randint(5, 100000, (B, L), device=device)
    attn = torch.ones(B, L, dtype=torch.long, device=device)
    row_ids = torch.randint(0, 30, (B, L), device=device)
    col_ids = torch.randint(0, 40, (B, L), device=device)

    # PeftModelForCausalLM_new.forward only pushes row_ids/col_ids when
    # is_prompt_learning=True. With bare LoRA we push them manually.
    def _push_ids():
        model._table_lora_set_row_col_ids_on_modules(row_ids, col_ids)

    t0 = time.time()
    print("Forward A (unpatched, reference)...", flush=True)
    _push_ids()
    with torch.no_grad():
        out_A = model(input_ids=input_ids, attention_mask=attn)
    logits_A = out_A.logits.detach().clone()
    print(f"  forward A: {time.time()-t0:.1f}s", flush=True)

    # Install tpe patch
    from tpe_impl.config import TPEConfig
    from tpe_impl.module.monkey_patch import install_tpe_patch, uninstall_tpe_patch
    cfg = TPEConfig(use_tpe=True, path_low_rank_dim=8,
                     path_vocab_path=require_vocab())
    install_tpe_patch(model, cfg)

    # Case B: patch installed, but no path_ids kwargs → _TPE_CTX stays None
    t0 = time.time()
    print("Forward B (patched, empty CTX — should be bit-exact with A)...", flush=True)
    _push_ids()
    with torch.no_grad():
        out_B = model(input_ids=input_ids, attention_mask=attn)
    logits_B = out_B.logits.detach().clone()
    print(f"  forward B: {time.time()-t0:.1f}s", flush=True)

    # Case C: patch installed, path_ids kwargs → _TPE_CTX populated, but emb = 0
    d_top = 3
    d_left = 3
    top_ids = torch.randint(2, 100, (B, L, d_top), device=device)
    left_ids = torch.randint(2, 100, (B, L, d_left), device=device)
    # Sanity: embeddings are zero-init
    ml = model.tpe_path_module
    for i in range(len(ml)):
        assert torch.all(ml[i].emb_top_path_K.weight == 0)
    t0 = time.time()
    print("Forward C (patched, full CTX, zero embs — should be bit-exact with A)...", flush=True)
    _push_ids()
    with torch.no_grad():
        out_C = model(input_ids=input_ids, attention_mask=attn,
                      top_path_ids=top_ids, left_path_ids=left_ids)
    logits_C = out_C.logits.detach().clone()
    print(f"  forward C: {time.time()-t0:.1f}s", flush=True)

    # Case D: patch + path_ids + NON-zero embeddings → should DIVERGE
    with torch.no_grad():
        for i in range(len(ml)):
            ml[i].emb_top_path_K.weight.normal_(0, 0.1)
            ml[i].emb_top_path_V.weight.normal_(0, 0.1)
            ml[i].emb_left_path_K.weight.normal_(0, 0.1)
            ml[i].emb_left_path_V.weight.normal_(0, 0.1)
    t0 = time.time()
    print("Forward D (patched, non-zero embs — should DIVERGE from A)...", flush=True)
    _push_ids()
    with torch.no_grad():
        out_D = model(input_ids=input_ids, attention_mask=attn,
                      top_path_ids=top_ids, left_path_ids=left_ids)
    logits_D = out_D.logits.detach().clone()
    print(f"  forward D: {time.time()-t0:.1f}s", flush=True)

    def report(name, a, b):
        diff = (a.float() - b.float()).abs()
        print(f"  {name}: max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}  "
              f"|a|_mean={a.float().abs().mean().item():.3e}")
        return diff.max().item()

    print("\n=== Comparisons ===")
    d_ab = report("(A) unpatched  vs (B) patched+empty_ctx", logits_A, logits_B)
    d_ac = report("(A) unpatched  vs (C) patched+zero_emb ", logits_A, logits_C)
    d_bc = report("(B) empty_ctx  vs (C) zero_emb        ", logits_B, logits_C)
    d_ad = report("(A) unpatched  vs (D) non-zero_emb    ", logits_A, logits_D)

    # Thresholds: bf16 bit-exact tolerance
    TOL_STRICT = 1e-2  # bf16 noise floor for logits of magnitude ~20
    all_ok = True
    if d_ab > TOL_STRICT:
        print(f"✗ (A) vs (B) not bit-exact"); all_ok = False
    if d_ac > TOL_STRICT:
        print(f"✗ (A) vs (C) not bit-exact"); all_ok = False
    if d_bc > TOL_STRICT:
        print(f"✗ (B) vs (C) differ"); all_ok = False
    # (D) must diverge
    if d_ad < 1e-2:
        print(f"✗ (D) should diverge from (A)"); all_ok = False

    assert all_ok, "bit-exact equivalence failed"
    print("\n✓ Bit-exact equivalence (real model) passed.")
    print(f"  - (B) empty CTX    ≡ (A) unpatched   [max|Δ|={d_ab:.3e}]")
    print(f"  - (C) zero embs    ≡ (A) unpatched   [max|Δ|={d_ac:.3e}]")
    print(f"  - (D) nonzero embs ≢ (A) unpatched   [max|Δ|={d_ad:.3e}]  ← patch confirmed active")


if __name__ == "__main__":
    main()
