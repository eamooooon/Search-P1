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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.trajectory_dump import _append_trajectory_dump
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 structure_format_score=0.,
                 final_format_score=0.,
                 retrieval_score=0.,
                 format_score=0.,
                 path_match_strategy="lexical",
                 require_search_for_format=False,
                 max_plan_steps=None,
                 max_reference_steps=None,
                 self_consistency_weight=0.0,
                 trajectory_dump_path=None,
                 trajectory_dump_limit=0,
                 trajectory_dump_split=None) -> None:
        qa_em_format.validate_path_match_strategy(path_match_strategy)
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.path_match_strategy = path_match_strategy
        self.require_search_for_format = require_search_for_format
        self.max_plan_steps = max_plan_steps
        self.max_reference_steps = max_reference_steps
        self.self_consistency_weight = self_consistency_weight
        self.trajectory_dump_path = trajectory_dump_path
        self.trajectory_dump_limit = int(trajectory_dump_limit or 0)
        self.trajectory_dump_split = trajectory_dump_split
        self._trajectory_dump_count = 0

    def _should_dump_trajectory(self):
        if not self.trajectory_dump_path:
            return False
        if self.trajectory_dump_limit == 0:
            return False
        return self.trajectory_dump_limit < 0 or self._trajectory_dump_count < self.trajectory_dump_limit

    def _dump_trajectory(self, *, solution_str, ground_truth, data_source, components, prompt=None, extra_info=None):
        if not self._should_dump_trajectory():
            return
        dump_split = self.trajectory_dump_split
        dump_index = self._trajectory_dump_count
        if isinstance(extra_info, dict):
            dump_split = extra_info.get("split", dump_split)
            dump_index = extra_info.get("index", dump_index)
        track_a_components = {
            key: components[key]
            for key in (
                "track_a_bonus",
                "self_consistency_weight",
                "self_consistency",
                "self_r_planner",
                "self_n_plan",
                "self_n_actions",
                "self_n_exec",
            )
            if key in components
        }
        track_b_components = {
            key: components[key]
            for key in (
                "reference_alignment",
                "ref_available",
                "ref_n_steps",
                "ref_n_actions",
                "ref_n_covered",
            )
            if key in components
        }
        _append_trajectory_dump(
            self.trajectory_dump_path,
            solution_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            split=dump_split,
            index=dump_index,
            track_a=track_a_components,
            track_b=track_b_components,
            prompt=prompt,
            extra_info=extra_info,
            reward_components=components,
        )
        self._trajectory_dump_count += 1

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        reward_components = {
            "base_score": [],
            "has_search": [],
            "effective_structure_format": [],
            "effective_retrieval": [],
            "track_a_bonus": [],
            "self_consistency_weight": [],
            "self_consistency": [],
            "self_r_planner": [],
            "self_n_plan": [],
            "self_n_actions": [],
            "self_n_exec": [],
            "reference_alignment": [],
            "ref_available": [],
            "ref_n_steps": [],
            "ref_n_actions": [],
            "ref_n_covered": [],
            "final_score": [],
        }

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            prompt = data_item.non_tensor_batch.get('prompt')
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            compute_score_fn = _select_rm_score_fn(data_source)

            if compute_score_fn is qa_em_format.compute_score_em:
                components = qa_em_format.compute_score_components(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    structure_format_score=self.structure_format_score,
                    final_format_score=self.final_format_score,
                    retrieval_score=self.retrieval_score,
                    format_score=self.format_score,
                    path_match_strategy=self.path_match_strategy,
                    require_search_for_format=self.require_search_for_format,
                    max_plan_steps=self.max_plan_steps,
                    max_reference_steps=self.max_reference_steps,
                    self_consistency_weight=self.self_consistency_weight,
                )
                score = components["final_score"]
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth,
                                         structure_format_score=self.structure_format_score,
                                         final_format_score=self.final_format_score,
                                         retrieval_score=self.retrieval_score,
                                         format_score=self.format_score,
                                         path_match_strategy=self.path_match_strategy,
                                         require_search_for_format=self.require_search_for_format,
                                         max_reference_steps=self.max_reference_steps,
                                         self_consistency_weight=self.self_consistency_weight)
                components = {
                    "base_score": score,
                    "has_search": 0.0,
                    "effective_structure_format": 1.0,
                    "effective_retrieval": 1.0,
                    "track_a_bonus": 0.0,
                    "self_consistency_weight": self.self_consistency_weight,
                    "self_consistency": 0.0,
                    "self_r_planner": 0.0,
                    "self_n_plan": 0.0,
                    "self_n_actions": 0.0,
                    "self_n_exec": 0.0,
                    "reference_alignment": 0.0,
                    "ref_available": 0.0,
                    "ref_n_steps": 0.0,
                    "ref_n_actions": 0.0,
                    "ref_n_covered": 0.0,
                    "final_score": score,
                }

            reward_tensor[i, valid_response_length - 1] = score
            for key, value in components.items():
                reward_components[key].append(float(value))
            self._dump_trajectory(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                data_source=data_source,
                components=components,
                prompt=prompt,
                extra_info=extra_info,
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.info("Decoded reward sample for %s:\n%s", data_source, sequences_str)

        data.meta_info["reward_components"] = reward_components
        return reward_tensor


def _reward_manager_kwargs(config):
    path_match_strategy = getattr(config.reward_model, "path_match_strategy", "lexical")
    qa_em_format.validate_path_match_strategy(path_match_strategy)
    return {
        "structure_format_score": config.reward_model.structure_format_score,
        "final_format_score": config.reward_model.final_format_score,
        "retrieval_score": config.reward_model.retrieval_score,
        "path_match_strategy": path_match_strategy,
        "require_search_for_format": getattr(config.reward_model, "require_search_for_format", False),
        "max_plan_steps": getattr(config.reward_model, "max_plan_steps", None),
        "max_reference_steps": getattr(config.reward_model, "max_reference_steps", None),
        "self_consistency_weight": getattr(config.reward_model, "self_consistency_weight", 0.0),
        "trajectory_dump_path": getattr(config.reward_model, "trajectory_dump_path", None),
        "trajectory_dump_limit": getattr(config.reward_model, "trajectory_dump_limit", 0),
    }


import ray
import hydra
import os


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        env_vars = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
        env_vars.update({
            key: value
            for key in ['SWANLAB_API_KEY', 'SWANLAB_MODE', 'SWANLAB_WORKSPACE', 'SWANLAB_RUN_ID', 'SWANLAB_RESUME']
            if (value := os.environ.get(key))
        })
        ray.init(runtime_env={'env_vars': env_vars})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    reward_config = _reward_manager_kwargs(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, trajectory_dump_split="train", **reward_config)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, trajectory_dump_split="val", **reward_config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
