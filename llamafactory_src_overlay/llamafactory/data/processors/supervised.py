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
from .processor_utils import greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_supervised_example(
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
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    if getattr(data_args, "emb_lora", False):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "row_ids": [], "col_ids": []}
        ori_tokenizer = deepcopy(tokenizer)
        # Paper's special tokens. Order matters: must match PromptEncoder virtual token indices.
        # [TAB] also serves as the placeholder for the table region inside the prompt.
        tokenizer.add_tokens(["[TAB]", "[ROW]", "[CELL]"], special_tokens=True)
    else:
        model_inputs = defaultdict(list)

    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        if getattr(data_args, "emb_lora", False):
            if len(examples["_prompt"][i]) != 1:
                raise ValueError(f'prompt should have only one turn')

            pattern = r"/\*\n(.*?)\n\*/"
            def replace_and_extract(text):
                matches = re.findall(pattern, text, re.DOTALL)
                replaced_text = re.sub(pattern, "/*\n[TAB]\n*/", text, flags=re.DOTALL)
                return replaced_text, matches

            prompt_content = examples["_prompt"][i][0]['content']
            examples["_prompt"][i][0]['content'], table_texts = replace_and_extract(prompt_content)

        input_ids, labels = _encode_supervised_example(
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
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        if not getattr(data_args, "emb_lora", False):
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
        elif not table_texts or len(table_texts) == 0:
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["row_ids"].append([0] * len(input_ids))
            model_inputs["col_ids"].append([0] * len(input_ids))
        elif "[ROW]" in table_texts[0] and "[CELL]" in table_texts[0]:
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                # Paper's table format: "[TAB] [ROW] [CELL] h1 [CELL] h2 [ROW] [CELL] v1 [CELL] v2"
                # We assign row_id=0 to the [TAB] marker, header row gets row_id=0,
                # then data rows get row_id=1,2,...
                # Within each row: row prefix ([ROW] + leading whitespace) gets col_id=0,
                # cells get col_id=1,2,3,...
                table_token_ids = []
                row_ids = []
                col_ids = []

                # Split by [ROW]: parts[0] is everything before the first [ROW]
                # (typically "[TAB] " — the table marker)
                parts = table_text.split("[ROW]")
                prefix_text = parts[0]
                if prefix_text:
                    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                    table_token_ids.extend(prefix_ids)
                    row_ids.extend([0] * len(prefix_ids))
                    col_ids.extend([0] * len(prefix_ids))

                for row_idx, row_text in enumerate(parts[1:]):
                    row_id = row_idx  # header row is row 0, then 1,2,...
                    # Within a row, split by [CELL]
                    # cell_parts[0] is whatever is between [ROW] and the first [CELL]
                    # (typically just whitespace).
                    cell_parts = row_text.split("[CELL]")
                    row_pre_ids = tokenizer.encode("[ROW]" + cell_parts[0], add_special_tokens=False)
                    table_token_ids.extend(row_pre_ids)
                    row_ids.extend([row_id] * len(row_pre_ids))
                    col_ids.extend([0] * len(row_pre_ids))

                    for col_idx, cell_text in enumerate(cell_parts[1:]):
                        col_id = col_idx + 1  # cells start at col 1
                        cell_ids = tokenizer.encode("[CELL]" + cell_text, add_special_tokens=False)
                        table_token_ids.extend(cell_ids)
                        row_ids.extend([row_id] * len(cell_ids))
                        col_ids.extend([col_id] * len(cell_ids))

                if input_ids.count(tokenizer.convert_tokens_to_ids("[TAB]")) == 0:
                    break
                table_insert_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[TAB]"))
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                if data_args.train_on_prompt:
                    labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
                else:
                    labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]
            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if len(input_ids) != len(row_ids_total) or len(input_ids) != len(col_ids_total):
                raise ValueError(f'input_ids {len(input_ids)}, row_ids {len(row_ids_total)}, col_ids {len(col_ids_total)} should have the same length')
        else:  # markdown format
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                table_token_ids = []
                col_ids = []
                row_ids = []
                row_id = 1
                col_id = 0
                for row_text in table_text.split("\n"):
                    col_id = 0
                    if set(row_text) == set(["|","-",":"]):
                        row_id -= 1
                        token_ids = tokenizer.encode(row_text+"\n")[1:]
                        table_token_ids.extend(token_ids)
                        row_ids.extend([row_id] * len(token_ids))
                        col_ids.extend([0] * len(token_ids))
                        row_id += 1
                    else:
                        row_pre_ids = tokenizer.encode("|"+"|".join(row_text.split("|")[:2])+"|")[1:]
                        table_token_ids.extend(row_pre_ids)
                        row_ids.extend([row_id] * len(row_pre_ids))
                        col_ids.extend([0] * len(row_pre_ids))
                        col_id += 1
                        for col_text in row_text.split("|")[2:]:
                            col_cell_ids = tokenizer.encode(col_text+"|")[1:]
                            table_token_ids.extend(col_cell_ids)
                            row_ids.extend([row_id] * len(col_cell_ids))
                            col_ids.extend([col_id] * len(col_cell_ids))
                            col_id += 1
                        row_id += 1

                if input_ids.count(tokenizer.convert_tokens_to_ids("[TAB]")) == 0:
                    break
                table_insert_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[TAB]"))
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                if data_args.train_on_prompt:
                    labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
                else:
                    labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]
            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if len(input_ids) != len(row_ids_total) or len(input_ids) != len(col_ids_total):
                raise ValueError(f'input_ids {len(input_ids)}, row_ids {len(row_ids_total)}, col_ids {len(col_ids_total)} should have the same length')

    if getattr(data_args, "emb_lora", False):
        tokenizer = ori_tokenizer
    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")
