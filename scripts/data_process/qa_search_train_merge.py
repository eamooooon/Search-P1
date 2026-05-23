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


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
Before any search, output exactly one complete plan block at the beginning. The planner contains numbered Step N: Search ... search-intent steps. Each step is one executable search goal, not necessarily the exact final query. If a later search depends on an unknown intermediate result, use placeholders like [identified actor], [identified film], or [target entity]. Do not list fallback branches, year-by-year searches, episode-by-episode searches, or long exhaustive lists. Keep the plan short and executable. \
After the plan, each action turn must contain one reasoning block followed immediately by either one tool_call block for search or one answer block for the final answer. \
The text inside tool_call must be only a plain search query. It must not contain a query prefix, search(...), any tag, a tool name, JSON/function-call text, a URL, or tool_response text. \
The assistant must never output <tool_response>. <tool_response> is returned only by the environment after a valid <tool_call>. When searching, assistant output must stop immediately after </tool_call> and wait for the environment. Do not invent observations or documents. \
Never output <query>, </query>, <tool_query>, </tool_query>, <search>, <think>, <information>, /query, tool_call: search, tool_response:, or JSON/function-call style tool calls. These are invalid. \
Use role-separated Search-P1 turns like this. Assistant output: <plan>\nStep 1: Search That Touch of Mink cast to identify the relevant actress.\nStep 2: Search [identified actress] role in The Honeymooners.\n</plan>\n<reasoning>I need to identify the actress from the film cast.</reasoning>\n<tool_call>That Touch of Mink cast</tool_call>\nEnvironment returns: <tool_response>Doc 1(Title: That Touch of Mink) The cast includes Joyce Randolph.</tool_response>\nAssistant output: <reasoning>Now I need her role in The Honeymooners.</reasoning>\n<tool_call>Joyce Randolph The Honeymooners role</tool_call>\nEnvironment returns: <tool_response>Doc 2(Title: Joyce Randolph) Joyce Randolph played Trixie Norton.</tool_response>\nAssistant output: <reasoning>The evidence is sufficient, so I can answer.</reasoning>\n<answer>Trixie Norton</answer>\nQuestion: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
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

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
