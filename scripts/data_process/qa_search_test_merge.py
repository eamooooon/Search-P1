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
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
from reference_steps import load_reference_steps, lookup_reference_steps


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
Before any search, output exactly one complete plan block at the beginning. The plan must contain numbered lines in the form Step N: Search ... and should cover the full search strategy before execution starts. \
After the plan, each action turn must contain one reasoning block followed immediately by either one tool_call block for search or one answer block for the final answer. \
The text inside tool_call must be only a plain search query. It must not contain a query prefix, search(...), any tag, a tool name, JSON/function-call text, a URL, or tool_response text. \
Never output <query>, </query>, <tool_query>, </tool_query>, <search>, <think>, <information>, /query, tool_call: search, tool_response:, or JSON/function-call style tool calls. These are invalid. \
Use one clean Search-P1 format like this: <plan>\nStep 1: Search Albert Einstein birthplace.\nStep 2: Search Albert Einstein Nobel Prize year.\n</plan>\n<reasoning>I need evidence for the birthplace.</reasoning>\n<tool_call>Albert Einstein birthplace</tool_call>\n<tool_response>Doc 1(Title: Example) Albert Einstein was born in Ulm.</tool_response>\n<reasoning>The evidence is sufficient, so I can answer.</reasoning>\n<answer>Ulm</answer>\nQuestion: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


def output_features():
    return datasets.Features({
        "data_source": datasets.Value("string"),
        "prompt": [{
            "role": datasets.Value("string"),
            "content": datasets.Value("string"),
        }],
        "ability": datasets.Value("string"),
        "reward_model": {
            "style": datasets.Value("string"),
            "ground_truth": {
                "target": datasets.Sequence(datasets.Value("string")),
                "reference_steps": datasets.Sequence(datasets.Value("string")),
            },
        },
        "extra_info": {
            "split": datasets.Value("string"),
            "index": datasets.Value("int64"),
        },
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--reference_steps_file', default=None)
    parser.add_argument('--max_reference_steps', type=int, default=None)

    args = parser.parse_args()
    references = load_reference_steps(
        args.reference_steps_file,
        max_reference_steps=args.max_reference_steps,
    )

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        if data_source != 'strategyqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="/home/peterjin/mnt/data/strategyqa/test_correct.jsonl")

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                reference_steps = lookup_reference_steps(
                    references,
                    data_source=data_source,
                    split=split,
                    index=idx,
                    question=example['question'],
                )
                solution = {
                    "target": example['golden_answers'],
                    "reference_steps": reference_steps,
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(
            function=make_map_fn('test'),
            with_indices=True,
            remove_columns=test_dataset.column_names,
            features=output_features(),
        )
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
