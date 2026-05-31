# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""In-memory prompt/response SFT dataset compatible with the local FSDP SFT trainer."""

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


def _series_to_item(value):
    while isinstance(value, (pd.Series, np.ndarray)) and len(value) == 1:
        value = value[0]
    return value


def _normalize_key(key):
    if isinstance(key, (tuple, list, ListConfig)):
        return list(key)
    return [key]


def _normalize_dict_keys(keys):
    if keys is None:
        return []
    if isinstance(keys, (tuple, list, ListConfig)):
        return list(keys)
    return [keys]


class SFTDataset(Dataset):
    """Load verl-style prompt/response parquet files for supervised fine-tuning.

    The output follows the contract expected by ``FSDPSFTTrainer``:
    ``input_ids``, ``attention_mask``, ``position_ids``, and ``loss_mask`` are
    fixed-length tensors. ``loss_mask`` masks prompt tokens and padding so the
    loss is applied only to the assistant response span.
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str], ListConfig],
        tokenizer: PreTrainedTokenizer,
        prompt_key="prompt",
        prompt_dict_keys=None,
        response_key="response",
        response_dict_keys=None,
        max_length=1024,
        truncation="error",
        cache_dir="~/.cache/verl/sft",
    ):
        if truncation not in ["error", "left", "right"]:
            raise ValueError(f"Unsupported truncation method: {truncation}")
        if not isinstance(parquet_files, (list, tuple, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.tokenizer = tokenizer
        self.prompt_key = _normalize_key(prompt_key)
        self.prompt_dict_keys = _normalize_dict_keys(prompt_dict_keys)
        self.response_key = _normalize_key(response_key)
        self.response_dict_keys = _normalize_dict_keys(response_dict_keys)
        self.max_length = max_length
        self.truncation = truncation
        self.cache_dir = cache_dir

        self._download()
        self._read_files()

    def _download(self):
        for index, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[index] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _extract_column(self, dataframe: pd.DataFrame, column_key, dict_keys):
        values = dataframe[column_key]
        for key in dict_keys:
            values = values.apply(lambda item: _series_to_item(item)[key], axis=1)
        if isinstance(values, pd.DataFrame):
            values = values.squeeze()
        return values.tolist()

    def _read_files(self):
        dataframes = [pd.read_parquet(parquet_file) for parquet_file in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f"dataset len: {len(self.dataframe)}")

        self.prompts = self._extract_column(self.dataframe, self.prompt_key, self.prompt_dict_keys)
        self.responses = self._extract_column(self.dataframe, self.response_key, self.response_dict_keys)

    def __len__(self):
        return len(self.prompts)

    def _format_prompt(self, prompt):
        if isinstance(prompt, list):
            chat = prompt
        else:
            chat = [{"role": "user", "content": str(prompt)}]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return chat[0]["content"]

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response = str(self.responses[item])
        prompt_text = self._format_prompt(prompt)
        eos_token = self.tokenizer.eos_token or ""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        prompt_ids_output = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        response_ids_output = self.tokenizer(response + eos_token, return_tensors="pt", add_special_tokens=False)

        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            padded_input_ids = torch.ones(size=(pad_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(pad_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids), dim=-1)
            attention_mask = torch.cat((attention_mask, padded_attention_mask), dim=-1)
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")

        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
