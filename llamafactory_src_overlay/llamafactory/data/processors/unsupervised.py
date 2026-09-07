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

import re
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_unsupervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int]]:
    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    messages = template.mm_plugin.process_messages(messages, images, videos, processor)
    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, images, videos, tokenizer, processor)
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), cutoff_len)
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    return input_ids, labels


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    if getattr(data_args, "emb_lora", False):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "row_ids": [], "col_ids": []}
        ori_tokenizer = deepcopy(tokenizer)
        # Same special tokens as supervised path; order MUST match prompt encoder local indices.
        tokenizer.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)
    else:
        model_inputs = defaultdict(list)

    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        if getattr(data_args, "emb_lora", False):
            if len(examples["_prompt"][i]) != 1:
                raise ValueError("emb_lora unsupervised path expects single-turn prompt")

            pattern = r"/\*\n(.*?)\n\*/"
            def replace_and_extract(text):
                matches = re.findall(pattern, text, re.DOTALL)
                replaced_text = re.sub(pattern, "/*\n[TAB]\n*/", text, flags=re.DOTALL)
                return replaced_text, matches

            prompt_content = examples["_prompt"][i][0]['content']
            examples["_prompt"][i][0]['content'], table_texts = replace_and_extract(prompt_content)

        input_ids, labels = _encode_unsupervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )

        if not getattr(data_args, "emb_lora", False):
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            continue

        # ===== emb_lora path: parse [ROW]/[CELL] and splice into input_ids =====
        # Same logic as the supervised processor; produces row_ids / col_ids that align with input_ids.
        if not table_texts or len(table_texts) == 0 or "[ROW]" not in table_texts[0] or "[CELL]" not in table_texts[0]:
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["row_ids"].append([0] * len(input_ids))
            model_inputs["col_ids"].append([0] * len(input_ids))
            continue

        row_ids_total = [0] * len(input_ids)
        col_ids_total = [0] * len(input_ids)
        for table_text in table_texts:
            table_token_ids = []
            row_ids = []
            col_ids = []

            parts = table_text.split("[ROW]")
            prefix_text = parts[0]
            if prefix_text:
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                table_token_ids.extend(prefix_ids)
                row_ids.extend([0] * len(prefix_ids))
                col_ids.extend([0] * len(prefix_ids))

            for row_idx, row_text in enumerate(parts[1:]):
                row_id = row_idx
                cell_parts = row_text.split("[CELL]")
                row_pre_ids = tokenizer.encode("[ROW]" + cell_parts[0], add_special_tokens=False)
                table_token_ids.extend(row_pre_ids)
                row_ids.extend([row_id] * len(row_pre_ids))
                col_ids.extend([0] * len(row_pre_ids))

                for col_idx, cell_text in enumerate(cell_parts[1:]):
                    col_id = col_idx + 1
                    cell_ids = tokenizer.encode("[CELL]" + cell_text, add_special_tokens=False)
                    table_token_ids.extend(cell_ids)
                    row_ids.extend([row_id] * len(cell_ids))
                    col_ids.extend([col_id] * len(cell_ids))

            if input_ids.count(tokenizer.convert_tokens_to_ids("[TAB]")) == 0:
                break
            table_insert_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[TAB]"))
            row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx + 1:]
            col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx + 1:]
            input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx + 1:]

        model_inputs["row_ids"].append(row_ids_total)
        model_inputs["col_ids"].append(col_ids_total)
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if len(input_ids) != len(row_ids_total) or len(input_ids) != len(col_ids_total):
            raise ValueError(
                f'input_ids {len(input_ids)}, row_ids {len(row_ids_total)}, col_ids {len(col_ids_total)} should have the same length'
            )

    if getattr(data_args, "emb_lora", False):
        tokenizer = ori_tokenizer
    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
