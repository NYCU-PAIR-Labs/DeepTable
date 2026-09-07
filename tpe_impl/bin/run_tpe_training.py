"""TPE training entry point (zero-intrusion wrapper around llamafactory-cli).

Usage:
  python -m tpe_impl.bin.run_tpe_training <yaml_path>

What it does:
  1. Read the yaml; if `use_tpe: true` →
  2. Monkey-patch `llamafactory.train.sft.workflow.load_model` so that after
     the model is built (with Peft + TableLoRA), we call `install_tpe_patch(...)`
     and (at predict time) `load_tpe_state(...)` from each adapter dir.
  3. Monkey-patch DataCollatorForSeq2Seq.__call__ to handle top_path_ids/left_path_ids (3D padding).
  4. Hijack the dataset processor: when `use_tpe: true` AND `emb_lora: true`,
     swap `preprocess_supervised_dataset` → `preprocess_tpe_supervised_dataset`.
  5. Patch BOTH `workflow.run_sft` AND `tuner.run_sft` so the run_sft wrapper
     actually fires regardless of which module's binding is resolved — this is
     what makes `data_args.use_tpe=True` get set before preprocess runs, and
     ensures the TPECheckpointCallback actually reaches the Trainer.
  6. If `tpe_lr_multiplier != 1.0`, patch CustomSeq2SeqTrainer.create_optimizer
     to put tpe params into their own param group with scaled lr (mirrors SAB's
     proven mechanism for tiny-params-need-big-lr).
  7. Register a Trainer callback that saves `tpe_modules.safetensors` + vocab json on save.
  8. Call LLaMA-Factory's main() normally (with a tpe-key-stripped yaml so LF's
     HfArgumentParser doesn't reject unknown fields).

If yaml does NOT have `use_tpe: true`, this script is equivalent to `llamafactory-cli train <yaml>`.

ZERO-INTRUSION: does not modify any file under `llamafactory_src_overlay/` or
`table_preprocess/`. All patching happens at runtime, all code lives in
`tpe_impl/`. When use_tpe=False the patches are no-op.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml as _yaml

# Ensure tpe_impl / deeptable_paths are importable (the user runs this as
# `python -m ...`, so path setup should already be fine, but belt-and-suspenders)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deeptable_paths import LF_SRC  # noqa: E402

# Also ensure `import llamafactory` resolves. When LLaMA-Factory was installed
# with `pip install -e .` this is redundant; when it was only cloned (see
# DEEPTABLE_LF_ROOT) it is what makes the import work.
if LF_SRC.is_dir() and str(LF_SRC) not in sys.path:
    sys.path.insert(0, str(LF_SRC))

os.environ.setdefault("TABLE_LORA_ENABLED", "1")


def _resolve_tpe_path_module(model):
    """Walk model / model.module / model.base_model / … looking for tpe_path_module.

    Trainer callbacks sometimes receive an Accelerate-wrapped or otherwise
    re-wrapped model where `hasattr(model, "tpe_path_module")` is False at the
    outer level even though it's present deeper. Returns the (owner, module)
    pair, or (None, None) if not found anywhere in the wrapper chain.
    """
    seen = set()
    cur = model
    for _ in range(8):  # bounded
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "tpe_path_module"):
            return cur, cur.tpe_path_module
        # common unwrap attrs
        for attr in ("module", "base_model", "model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and id(nxt) not in seen:
                cur = nxt
                break
        else:
            break
    return None, None


def install_all_patches(yaml_path: str):
    """Install tpe patches IF the yaml enables use_tpe. Returns the TPEConfig."""
    from tpe_impl.config import TPEConfig
    cfg = TPEConfig.from_yaml(yaml_path)

    if not cfg.use_tpe:
        print(f"[tpe-bootstrap] yaml {yaml_path}: use_tpe is False/absent; "
              "no patches installed.", flush=True)
        return cfg

    print(f"[tpe-bootstrap] Installing tpe patches for yaml: {yaml_path}", flush=True)
    vocab_meta = cfg.load_vocab()
    unk_id = vocab_meta["unk_id"]
    digest = cfg.vocab_file_digest
    print(f"[tpe-bootstrap]   vocab_size={cfg.vocab_size}  digest={digest}  "
          f"tpe_lr_mult={cfg.tpe_lr_multiplier}", flush=True)

    # 1. (deferred) The collator patch MUST run AFTER TableLoRA's load_table_lora()
    #    because that function also patches DataCollatorForSeq2Seq.__call__
    #    (table_lora.py:1585) and would otherwise overwrite our chained patch.
    #    load_table_lora is called from tuner.run_exp (when emb_lora=True) BEFORE
    #    run_sft is invoked — so we install the tpe collator patch inside the
    #    _patched_run_sft wrapper below, not here.
    from tpe_impl.collator import install_tpe_collator_patch

    # 2. Dataset processor swap (tpe_supervised replaces supervised for use_tpe=True)
    import llamafactory.data.processors.supervised as _supervised
    from tpe_impl.processors.tpe_supervised import preprocess_tpe_supervised_dataset

    _orig_preprocess = _supervised.preprocess_supervised_dataset

    def _tpe_gated_preprocess(examples, template, tokenizer, processor, data_args):
        if getattr(data_args, "use_tpe", False):
            setattr(data_args, "path_vocab_path", cfg.path_vocab_path)
            return preprocess_tpe_supervised_dataset(examples, template, tokenizer, processor, data_args)
        return _orig_preprocess(examples, template, tokenizer, processor, data_args)

    _supervised.preprocess_supervised_dataset = _tpe_gated_preprocess
    try:
        from llamafactory.data import preprocess as _preprocess_mod
        if hasattr(_preprocess_mod, "preprocess_supervised_dataset"):
            _preprocess_mod.preprocess_supervised_dataset = _tpe_gated_preprocess
    except ImportError:
        pass
    print("[tpe-bootstrap]   preprocess_supervised_dataset monkey-patched", flush=True)

    # 3. Wrap load_model: post-build, install_tpe_patch (train) AND load_tpe_state
    #    for each adapter dir (predict).
    from llamafactory.train.sft import workflow as _workflow
    from tpe_impl.module import install_tpe_patch, load_tpe_state

    _orig_load_model = _workflow.load_model

    def _patched_load_model(*args, **kwargs):
        # Signature: load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
        _model_args = args[1] if len(args) >= 2 else kwargs.get("model_args")
        model = _orig_load_model(*args, **kwargs)
        install_tpe_patch(model, cfg)
        # Belt-and-suspenders: force requires_grad=True on every tpe param.
        # (PEFT freezes the base model; tpe params are added after and default
        # to requires_grad=True, but mirror SAB's loud-assert pattern.)
        for n, p in model.tpe_path_module.named_parameters():
            if not p.requires_grad:
                print(f"[tpe] forcing requires_grad=True on tpe_path_module.{n}", flush=True)
                p.requires_grad_(True)
        # Predict path: PEFT just reloaded a saved adapter from model_args.adapter_name_or_path.
        # PEFT only restores lora_* weights; our tpe_path_module is in a companion file.
        adapters = getattr(_model_args, "adapter_name_or_path", None) if _model_args else None
        if adapters:
            for ad in adapters:
                try:
                    load_tpe_state(model, ad, strict=True)
                except FileNotFoundError as e:
                    # Strict load fails loudly if the adapter dir predates the save fix;
                    # surface the error with a hint rather than silently running zero-init.
                    print(f"[tpe] ERROR: {e}", flush=True)
                    raise
        return model

    _workflow.load_model = _patched_load_model
    print("[tpe-bootstrap]   load_model wrapped — install_tpe_patch + load_tpe_state", flush=True)

    # 4. TPECheckpointCallback — saves on every checkpoint + at train end.
    #    Primary path: walk `model` arg through common unwrap attrs.
    #    Fallback: module-level `_TPE_MODULES` global set by install_tpe_patch.
    import transformers
    from tpe_impl.module import save_tpe_state
    from tpe_impl.module import monkey_patch as _tpe_mp  # for _TPE_MODULES global

    def _save_to(save_dir, model_arg):
        owner, module = _resolve_tpe_path_module(model_arg)
        if module is None:
            module = getattr(_tpe_mp, "_TPE_MODULES", None)
            if module is None:
                print(f"[tpe] save_to {save_dir}: no tpe_path_module found; "
                      "callback skipping.", flush=True)
                return
            class _Shim:
                pass
            owner = _Shim()
            owner.tpe_path_module = module
        save_tpe_state(owner, save_dir, cfg)

    class TPECheckpointCallback(transformers.TrainerCallback):
        def on_save(self, args, state, control, model=None, **kw):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                _save_to(ckpt_dir, model)

        def on_train_end(self, args, state, control, model=None, **kw):
            _save_to(args.output_dir, model)

    class TPEMagnitudeCallback(transformers.TrainerCallback):
        """Once per epoch, log tpe embedding magnitudes (sanity that training is
        actually moving the embeddings off zero-init). Low-frequency, low-cost."""
        def on_epoch_end(self, args, state, control, model=None, **kw):
            owner, tpe_module = _resolve_tpe_path_module(model)
            if tpe_module is None:
                return
            try:
                probes = {idx: tpe_module[idx].emb_top_path_K.weight
                          for idx in (0, 15, 29) if idx < len(tpe_module)}
                msg = "  ".join(
                    f"L{idx}={w.abs().max().item():.3e}" for idx, w in probes.items())
                print(f"[tpe] epoch {state.epoch:.2f} emb_top_path_K absmax: {msg}",
                      flush=True)
            except Exception as e:
                print(f"[tpe] magnitude probe failed: {e}", flush=True)

    # 5. Wrap BOTH workflow.run_sft AND tuner.run_sft. Patching only the
    #    workflow module is insufficient because tuner.py does
    #    `from .sft import run_sft` at import time, binding the name in
    #    tuner's own namespace. The call path that actually fires is
    #    `tuner.run_exp → run_sft(...)` which resolves via tuner.__dict__.
    from llamafactory.train.sft import workflow as _w
    import llamafactory.train.tuner as _tuner

    _orig_run_sft = _w.run_sft

    def _patched_run_sft(model_args, data_args, training_args, finetuning_args,
                         generating_args, callbacks=None):
        setattr(data_args, "use_tpe", True)
        setattr(data_args, "path_vocab_path", cfg.path_vocab_path)
        setattr(data_args, "path_low_rank_dim", cfg.path_low_rank_dim)
        # Install tpe collator patch HERE, after tuner.run_exp has already run
        # load_table_lora() — so our patch chains on top of TableLoRA's collator
        # patch rather than being overwritten by it. Idempotent (install checks
        # _INSTALLED flag).
        install_tpe_collator_patch(unk_id=unk_id)
        if callbacks is None:
            callbacks = []
        callbacks.append(TPECheckpointCallback())
        callbacks.append(TPEMagnitudeCallback())
        print(f"[tpe] _patched_run_sft fired: data_args.use_tpe=True, "
              f"TPECheckpointCallback registered (total callbacks={len(callbacks)})",
              flush=True)
        return _orig_run_sft(model_args, data_args, training_args, finetuning_args,
                             generating_args, callbacks)

    _w.run_sft = _patched_run_sft
    _tuner.run_sft = _patched_run_sft  # ← the one that actually gets called
    print("[tpe-bootstrap]   run_sft wrapped in BOTH workflow and tuner modules", flush=True)

    # 6. Optional: separate param group for tpe params with scaled lr
    #    (mirrors SAB's proven mechanism; no-op when tpe_lr_multiplier == 1.0)
    if cfg.tpe_lr_multiplier != 1.0:
        import llamafactory.train.sft.trainer as _sft_trainer
        if not getattr(_sft_trainer, "_tpe_create_optimizer_wrapped", False):
            _orig_create_optimizer = _sft_trainer.CustomSeq2SeqTrainer.create_optimizer

            def _create_optimizer_with_tpe_lr(self):
                result = _orig_create_optimizer(self)
                owner, tpe_module = _resolve_tpe_path_module(self.model)
                if tpe_module is None:
                    return result
                tpe_ids = {id(p) for p in tpe_module.parameters()}
                new_groups = []
                tpe_collected = []
                for group in self.optimizer.param_groups:
                    base = [p for p in group["params"] if id(p) not in tpe_ids]
                    tpe = [p for p in group["params"] if id(p) in tpe_ids]
                    if base:
                        g = dict(group)
                        g["params"] = base
                        new_groups.append(g)
                    if tpe:
                        g = dict(group)
                        g["params"] = tpe
                        g["lr"] = group["lr"] * cfg.tpe_lr_multiplier
                        g["initial_lr"] = g["lr"]  # cosine scheduler base
                        tpe_collected.extend(tpe)
                        new_groups.append(g)
                if tpe_collected:
                    self.optimizer.param_groups = new_groups
                    # Find a tpe group (one whose params include one of tpe_ids)
                    tpe_group_lr = None
                    for g in new_groups:
                        if any(id(p) in tpe_ids for p in g["params"]):
                            tpe_group_lr = g["lr"]
                            break
                    print(
                        f"[tpe] Separate param group: base_lr={self.args.learning_rate:.2e} "
                        f"tpe_lr={tpe_group_lr:.2e} (×{cfg.tpe_lr_multiplier}) "
                        f"tpe_params={sum(p.numel() for p in tpe_collected)}",
                        flush=True,
                    )
                return result

            _sft_trainer.CustomSeq2SeqTrainer.create_optimizer = _create_optimizer_with_tpe_lr
            _sft_trainer._tpe_create_optimizer_wrapped = True
            print(f"[tpe-bootstrap]   create_optimizer wrapped for tpe_lr_multiplier="
                  f"{cfg.tpe_lr_multiplier}", flush=True)

    return cfg


def _strip_tpe_keys_to_temp_yaml(yaml_path: str) -> str:
    """Write a copy of the yaml with tpe_* fields stripped, to a temp file.

    LLaMA-Factory's HfArgumentParser raises ValueError on unknown yaml keys in
    transformers 4.46.1. TPE fields are consumed by our wrapper (via TPEConfig)
    and re-injected onto data_args inside run_sft, so they're safe to strip here.
    """
    with open(yaml_path) as f:
        cfg_dict = _yaml.safe_load(f)
    for k in ("use_tpe", "path_vocab_path", "path_low_rank_dim",
              "tpe_lr_multiplier", "disable_2d_lora",
              "pool_mode", "tpe_top_only"):
        cfg_dict.pop(k, None)
    base = Path(yaml_path).stem
    fd, tmp_path = tempfile.mkstemp(prefix=f"lf_{base}_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        _yaml.safe_dump(cfg_dict, f, sort_keys=False)
    return tmp_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tpe_impl.bin.run_tpe_training <yaml_path>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    cfg = install_all_patches(yaml_path)

    if cfg.use_tpe:
        lf_yaml = _strip_tpe_keys_to_temp_yaml(yaml_path)
        print(f"[tpe-bootstrap]   sanitized yaml for LF: {lf_yaml}", flush=True)
    else:
        lf_yaml = yaml_path

    from llamafactory.cli import main as _lf_main
    sys.argv = ["llamafactory-cli", "train", lf_yaml] + sys.argv[2:]
    _lf_main()


if __name__ == "__main__":
    main()
