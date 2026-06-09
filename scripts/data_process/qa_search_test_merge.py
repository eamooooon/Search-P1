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
        prefix = f"""You are a meticulous Deep Research Agent. Answer the question by planning, searching, reading evidence, and giving a concise final answer.
## CRITICAL INSTRUCTIONS
1. Detailed Planning (<plan>):
- In the first turn, output exactly one complete plan block.
- Break the question into executable search-intent steps.
- Use numbered lines in the form: Step N: Search ...
- Focus on one sub-question at a time.
2. Step-by-Step Execution (<search>):
- After the plan, each assistant turn must contain one <think> block followed by either one <search> block or one <answer> block.
- Execute only one plain natural language search query per turn.
- After receiving <information>, use <think> to decide whether the evidence is sufficient or another search is needed.
3. Evidence Boundary:
- <information> is returned only by the environment after a valid <search>.
- When you finish </search>, stop and wait for the environment.
- Do not invent observations or documents.
4. Final Answer (<answer>):
- Output <answer> only when the necessary information is gathered.
- Keep the final answer concrete and short.
## CURRENT TASK
Question: {question}
"""
    elif template_type == 'medium':
        prefix = f"""You are a meticulous Deep Research Agent. Answer the question using the Search-R1 format.

Turn 1 must contain exactly one plan block followed by exactly one search action:
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
<think>Briefly explain the next action.</think>
<search>plain natural language search query</search>

After an information block, never write another <plan>. Each later turn must be exactly one think block followed by one action block:
<think>Briefly explain what to do next.</think>
<search>plain natural language search query</search>

or:
<think>The evidence is sufficient.</think>
<answer>short final answer</answer>

Hard rules:
- You must perform at least one <search> before any <answer>.
- Always close the final action with </search> or </answer>.
- A <think>...</think> block must immediately precede every <search> or <answer>.
- Use exactly one <plan> block total; repeating <plan> is invalid.
- In <plan>, placeholders like [identified actress] are allowed when a later search depends on an earlier result.
- Inside <search>, write only a concrete plain query, such as Arthur's Magazine founding date.
- Inside <search>, replace any plan placeholder with the concrete entity learned from <information>.
- Do not use unresolved placeholders in <search> or <answer>; the final <answer> must be concrete and short.
- When you finish </search>, stop. The environment will return <information>.

Example first turn:
<plan>
Step 1: Search That Touch of Mink cast to identify the relevant actress.
Step 2: Search [identified actress] The Honeymooners role.
</plan>
<think>I need to identify the actress from the film.</think>
<search>That Touch of Mink cast</search>

Example later turn after information:
<think>The result identifies Joyce Randolph, so I need her Honeymooners role.</think>
<search>Joyce Randolph The Honeymooners role</search>

Example final turn:
<think>The evidence is sufficient.</think>
<answer>Trixie Norton</answer>

Question: {question}
"""
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
    use_explicit_features = bool(args.reference_steps_file)

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
                }
                if use_explicit_features:
                    solution["reference_steps"] = reference_steps

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

        map_kwargs = {"function": make_map_fn('test'), "with_indices": True}
        if use_explicit_features:
            map_kwargs.update({
                "remove_columns": test_dataset.column_names,
                "features": output_features(),
            })
        test_dataset = test_dataset.map(**map_kwargs)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
