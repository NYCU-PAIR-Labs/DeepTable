# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if getattr(data_args, "emb_lora", False):
        import os
        os.environ["TABLE_LORA_ENABLED"] = "1"
        from llamafactory.table_lora.table_lora import load_table_lora, set_special_token_ids
        load_table_lora()

        # Pre-compute the [TAB] / [ROW] / [CELL] token ids using the same tokenizer
        # the trainer will use, then register them so PeftModel_new.forward can route
        # them through the prompt encoder. Order MUST match the PromptEncoder virtual
        # token indices: [TAB]=0, [ROW]=1, [CELL]=2.
        from transformers import AutoTokenizer
        _tok_for_ids = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        _tok_for_ids.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)
        _ids = [_tok_for_ids.convert_tokens_to_ids(t) for t in ["[TAB]", "[ROW]", "[CELL]"]]
        set_special_token_ids(_ids)

    # ------------------------------------------------------------------
    # SAB (Structural Attention Bias) — independent add-on to TableLoRA.
    # Lives entirely in llamafactory.sab and never touches table_lora.py.
    # See llamafactory/sab/sab.py for the design.
    # ------------------------------------------------------------------
    if getattr(data_args, "use_sab", False):
        if not getattr(data_args, "emb_lora", False):
            raise ValueError("use_sab=true requires emb_lora=true (SAB consumes row_ids/col_ids).")
        # SAB now patches BOTH LlamaSdpaAttention.forward and
        # LlamaAttention.forward (eager fallback), so we no longer need to
        # force eager. The model can use whatever attention backend it defaults
        # to (typically SDPA on torch>=2.1), preserving training speed.
        # Only flash_attention_2 is unsupported (SDPA and eager are both fine).
        if model_args.flash_attn == "fa2":
            logger.warning_rank0(
                "use_sab=true is incompatible with flash_attn='fa2'; "
                "falling back to 'auto' (SDPA)."
            )
            model_args.flash_attn = "auto"

        # Install the LlamaAttention monkey-patch up front so the patched
        # forward is in place by the time the first attention layer runs.
        # When _SAB_CTX["module"] is None, the patched forward is a no-op
        # and is bit-equivalent to upstream — proven by sab_sanity_check.py.
        from llamafactory.sab import (
            install_sab_attention_patch,
            load_sab,
            save_sab,
            load_sab_state,
        )
        install_sab_attention_patch()

        # Wrap llamafactory.train.sft.workflow.load_model so the model the
        # SFT workflow gets back already has SAB attached BEFORE the Trainer
        # is constructed. This is the *only* timing that works:
        #
        #   * on_init_end:   no `model` in kwargs (HF Trainer.__init__ source).
        #   * on_train_begin: fires AFTER create_optimizer_and_scheduler, so
        #                     SAB params would be missing from the optimizer
        #                     param group → no gradient updates.
        #
        # Wrapping load_model installs SAB while the model is still a plain
        # peft module, BEFORE Trainer ever sees it, so `model.parameters()`
        # already includes the 1920 SAB scalars when create_optimizer runs.
        # The same wrapper *also* loads SAB state from each adapter directory
        # so the inference path picks up the trained alpha matrices.
        import llamafactory.train.sft.workflow as _sft_wf
        if not getattr(_sft_wf, "_sab_load_model_wrapped", False):
            _orig_load_model = _sft_wf.load_model

            def _load_model_with_sab(*args, **kwargs):
                # signature: load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)
                _model_args = args[1] if len(args) >= 2 else kwargs.get("model_args")
                model = _orig_load_model(*args, **kwargs)
                load_sab(model)
                # Make absolutely sure the SAB params are trainable. PEFT
                # froze the base model in load_model, then unfroze its lora
                # params; SAB params are added afterwards so they default to
                # requires_grad=True, but assert it loudly in case future
                # PEFT versions walk the param tree differently.
                for n, p in model.sab_module.named_parameters():
                    if not p.requires_grad:
                        logger.warning_rank0(f"[sab] forcing requires_grad=True on sab_module.{n}")
                        p.requires_grad_(True)
                # Inference path: PEFT just reloaded a saved adapter from
                # `model_args.adapter_name_or_path`. PEFT itself only restores
                # LoRA + prompt_encoder weights; the trained sab_module lives
                # in our companion `sab_module.safetensors` next to the
                # adapter. Try to load it from each adapter dir.
                adapters = getattr(_model_args, "adapter_name_or_path", None) if _model_args else None
                if adapters:
                    for ad in adapters:
                        load_sab_state(model, ad, strict=False)
                return model

            _sft_wf.load_model = _load_model_with_sab
            _sft_wf._sab_load_model_wrapped = True
            logger.info_rank0("[sab] wrapped llamafactory.train.sft.workflow.load_model — SAB will be attached before Trainer init.")

        # ----------- callbacks: alpha monitor + checkpoint save ------------
        from transformers import TrainerCallback

        class SABMonitorCallback(TrainerCallback):
            """Print alpha_row.abs().mean() / alpha_col.abs().mean() every
            `every_n` steps so the user can track SAB learning during a long
            training run."""

            def __init__(self, every_n: int = 200):
                self.every_n = every_n

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if model is None or not hasattr(model, "sab_module"):
                    return
                step = state.global_step
                if step == 1 or step % self.every_n == 0:
                    ar = model.sab_module.alpha_row
                    ac = model.sab_module.alpha_col
                    print(
                        f"[sab-monitor] step={step:>5d}  "
                        f"alpha_row absmean={ar.abs().mean().item():.4e} absmax={ar.abs().max().item():.4e}  "
                        f"alpha_col absmean={ac.abs().mean().item():.4e} absmax={ac.abs().max().item():.4e}",
                        flush=True,
                    )

        class SABCheckpointCallback(TrainerCallback):
            """On every checkpoint save (and at end of training), write the
            SAB state to `sab_module.safetensors` next to the PEFT adapter.
            PEFT's save_pretrained does NOT capture our sab_module — verified
            by sab_save_load_test.py."""

            def on_save(self, args, state, control, model=None, **kwargs):
                if model is None or not hasattr(model, "sab_module"):
                    return
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if os.path.isdir(ckpt_dir):
                    save_sab(model, ckpt_dir)

            def on_train_end(self, args, state, control, model=None, **kwargs):
                if model is None or not hasattr(model, "sab_module"):
                    return
                save_sab(model, args.output_dir)

        callbacks.append(SABMonitorCallback(every_n=200))
        callbacks.append(SABCheckpointCallback())

        # ----------- separate param group for SAB (optional) ---------------
        # When SAB_LR_MULTIPLIER > 1, SAB's 1920 parameters get their own
        # optimizer param group with a higher lr. This solves the "alpha too
        # small" problem: the base lr (5e-6) is tuned for millions of LoRA
        # params, but SAB's tiny scalar array needs a much larger lr to
        # escape the 0-neighborhood. Controlled via env var so the same code
        # path works for both the default run (multiplier=1) and the 100x
        # experiment (multiplier=100).
        sab_lr_multiplier = float(os.environ.get("SAB_LR_MULTIPLIER", "1"))
        if sab_lr_multiplier != 1.0:
            import llamafactory.train.sft.trainer as _sft_trainer
            if not getattr(_sft_trainer, "_sab_create_optimizer_wrapped", False):
                _orig_create_optimizer = _sft_trainer.CustomSeq2SeqTrainer.create_optimizer

                def _create_optimizer_with_sab_lr(self):
                    result = _orig_create_optimizer(self)
                    if not hasattr(self.model, "sab_module"):
                        return result
                    sab_ids = {id(p) for p in self.model.sab_module.parameters()}
                    # Re-partition: pull SAB params out of existing groups,
                    # give them their own group with scaled lr.
                    new_groups = []
                    sab_collected = []
                    for group in self.optimizer.param_groups:
                        base = [p for p in group["params"] if id(p) not in sab_ids]
                        sab = [p for p in group["params"] if id(p) in sab_ids]
                        if base:
                            g = dict(group)
                            g["params"] = base
                            new_groups.append(g)
                        if sab:
                            g = dict(group)
                            g["params"] = sab
                            g["lr"] = group["lr"] * sab_lr_multiplier
                            g["initial_lr"] = g["lr"]  # so cosine scheduler sees the right base
                            sab_collected.extend(sab)
                            new_groups.append(g)
                    if sab_collected:
                        self.optimizer.param_groups = new_groups
                        sab_lr = new_groups[-1]["lr"]
                        print(
                            f"[sab] Separate param group: base_lr={self.args.learning_rate:.2e} "
                            f"sab_lr={sab_lr:.2e} (×{sab_lr_multiplier}) "
                            f"sab_params={sum(p.numel() for p in sab_collected)}",
                            flush=True,
                        )
                    return result

                _sft_trainer.CustomSeq2SeqTrainer.create_optimizer = _create_optimizer_with_sab_lr
                _sft_trainer._sab_create_optimizer_wrapped = True
                logger.info_rank0(f"[sab] wrapped create_optimizer for SAB lr multiplier={sab_lr_multiplier}")

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")


def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")
