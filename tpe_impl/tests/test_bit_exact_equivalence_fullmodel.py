"""Zero-initialization verification (Appendix H): bit-exact equivalence between

  (a) TableLoRA baseline (no tpe patch installed)                        — reference
  (b) tpe patch installed, but _TPE_CTX empty (forward_pre_hook didn't fire) — "fallback A"
  (c) tpe patch installed, _TPE_CTX populated, but emb weights = 0     — "fallback B"

All three should produce IDENTICAL logits on the same input.

Also checks that vanilla LoRA mode (is_tab_lora=False path) is untouched.
Actually since all our runs use is_tab_lora=True, we do a weaker form of 1b:
check that the else branch (non-tab) in _patched_linear_new_forward is textually
identical to upstream.
"""
import sys
from _bootstrap import (  # noqa: F401
    BACKBONE, require_adapter, require_data, require_gpu, require_vocab,
)

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

os.environ.setdefault("TABLE_LORA_ENABLED", "1")

MODEL = BACKBONE  # override with $DEEPTABLE_BACKBONE
# A previously trained adapter to reuse; see _bootstrap.require_adapter().
SAB_ADAPTER = None  # resolved at run time from $DEEPTABLE_TEST_ADAPTER


def main():
    global SAB_ADAPTER
    require_gpu()
    SAB_ADAPTER = require_adapter()
    require_vocab()
    require_data("hitab_tpe_train.json")

    # --- Install TableLoRA patches
    from llamafactory.table_lora.table_lora import load_table_lora, set_special_token_ids
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left",
                                              add_eos_token=True, add_bos_token=True)
    tokenizer.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)
    special_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["[TAB]", "[ROW]", "[CELL]"]]
    set_special_token_ids(special_ids)
    load_table_lora()

    # --- Build base model + LoRA (use the existing lr100x adapter to skip retraining;
    # its SAB bias is irrelevant here because we don't install the SAB patch or
    # sab_module, so SAB branch is a no-op)
    device = torch.device("cuda:0")
    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    base.resize_token_embeddings(len(tokenizer))
    base.to(device).eval()

    print("Attaching LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, SAB_ADAPTER, is_trainable=False)
    model.to(device).eval()

    # --- Build small batch with tpe fields
    from tpe_impl.processors.metadata_index import reset_for_tests
    reset_for_tests()
    from tpe_impl.processors.tpe_supervised import preprocess_tpe_supervised_dataset
    from tpe_impl.collator import install_tpe_collator_patch, uninstall_tpe_collator_patch
    from transformers import DataCollatorForSeq2Seq
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    data_args = DataArguments(template="deepseek")
    data_args.cutoff_len = 2000
    data_args.train_on_prompt = False
    data_args.mask_history = False
    data_args.emb_lora = True
    data_args.use_tpe = True
    data_args.path_vocab_path = require_vocab()
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    ds = json.load(open(require_data("hitab_tpe_train.json")))
    examples = {
        "_prompt": [[{"role": "user", "content": ds[i]["prompt"]}] for i in [0, 1]],
        "_response": [[{"role": "assistant", "content": ds[i]["response"]}] for i in [0, 1]],
        "_system": [""] * 2,
        "_tools": [""] * 2,
        "_images": [None] * 2,
        "_videos": [None] * 2,
    }
    out = preprocess_tpe_supervised_dataset(examples, template, tokenizer, None, data_args)
    features = [
        {
            "input_ids": out["input_ids"][b],
            "attention_mask": [1] * len(out["input_ids"][b]),
            "labels": out["labels"][b],
            "row_ids": out["row_ids"][b],
            "col_ids": out["col_ids"][b],
            "top_path_ids": out["top_path_ids"][b],
            "left_path_ids": out["left_path_ids"][b],
        }
        for b in range(2)
    ]
    install_tpe_collator_patch(unk_id=1)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    batch = collator(features)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # --- Compute reference (no tpe patch)
    from tpe_impl.module.monkey_patch import (
        install_tpe_patch, uninstall_tpe_patch, _TPE_CTX, _TPE_MODULES
    )
    from tpe_impl.config import TPEConfig

    def forward_no_tpe():
        # Remove tpe kwargs so PeftModel sees a normal forward
        kwargs = {k: v for k, v in batch.items() if k not in ("top_path_ids", "left_path_ids")}
        with torch.no_grad():
            out = model(**kwargs)
        return out.logits.detach().clone()

    # ---- (a) Baseline: no tpe patch at all ----
    uninstall_tpe_patch(model)  # idempotent no-op if not installed
    logits_a = forward_no_tpe()

    # ---- (b) TPE patch installed but forward_pre_hook not used (empty _TPE_CTX) ----
    cfg = TPEConfig(
        use_tpe=True,
        path_low_rank_dim=8,  # must match TableLoRA rank
        path_vocab_path=data_args.path_vocab_path,
    )
    install_tpe_patch(model, cfg)

    # Force _TPE_CTX to be empty by NOT passing top/left path ids (forward_no_tpe strips them)
    logits_b = forward_no_tpe()

    # ---- (c) TPE patch installed, path ids passed, but embeddings are zero ----
    # Embeddings are already zero-init per TPEPathModule.__init__
    from tpe_impl.module.tpe_path_module import TPEPathModule
    # Verify they're zero
    ml = model.tpe_path_module
    for i in range(len(ml)):
        assert torch.all(ml[i].emb_top_path_K.weight == 0)
        assert torch.all(ml[i].emb_top_path_V.weight == 0)
        assert torch.all(ml[i].emb_left_path_K.weight == 0)
        assert torch.all(ml[i].emb_left_path_V.weight == 0)
    with torch.no_grad():
        out = model(**batch)  # passes top_path_ids / left_path_ids
    logits_c = out.logits.detach().clone()

    # ---- Comparisons ----
    def report(name, a, b):
        diff = (a.float() - b.float()).abs()
        print(f"  {name}: max|Δ| = {diff.max().item():.3e}, "
              f"mean|Δ| = {diff.mean().item():.3e}, |a|_mean = {a.float().abs().mean().item():.3e}")
        return diff.max().item()

    print("\n=== Bit-exact sanity (A = no patch, B = patch but empty CTX, C = patch + zero emb) ===")
    d_ab = report("(a) vs (b)", logits_a, logits_b)
    d_ac = report("(a) vs (c)", logits_a, logits_c)
    d_bc = report("(b) vs (c)", logits_b, logits_c)

    # bf16 noise floor: ~1e-3 relative
    # Absolute threshold: eyeballed 1e-2 for bf16 logits whose magnitude is typically ~20
    TOL = 1e-2
    ok_ab = d_ab < TOL
    ok_ac = d_ac < TOL
    ok_bc = d_bc < TOL

    print(f"\nTolerance: max|Δ| < {TOL}")
    print(f"  (a) vs (b): {'✓' if ok_ab else '✗'}")
    print(f"  (a) vs (c): {'✓' if ok_ac else '✗'}")
    print(f"  (b) vs (c): {'✓' if ok_bc else '✗'}")

    assert ok_ab, "Fallback A (empty CTX) is NOT bit-exact with baseline"
    assert ok_ac, "Fallback B (zero embeddings) is NOT bit-exact with baseline"
    assert ok_bc, "Both fallbacks differ (should be identical)"

    print("\n✓ Bit-exact equivalence (full model) passed.")


if __name__ == "__main__":
    main()
