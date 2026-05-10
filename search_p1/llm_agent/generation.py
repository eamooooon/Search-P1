import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional, DefaultDict
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests  # type: ignore[import-untyped]

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: Optional[str] = None
    topk: int = 3

class LLMGenerationManager:
    PLAN_ACCEPTED_OBSERVATION = (
        "\nPlan accepted. Do not output <plan> again. Now output exactly one "
        "<reasoning>...</reasoning> followed by one <tool_call>...</tool_call> "
        "or <answer>...</answer>.\n"
    )

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at tool-call operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [self._truncate_at_first_action(resp) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _truncate_at_first_action(self, response: str) -> str:
        action_endings = ["</tool_call>", "</answer>"]
        candidates = [
            (response.find(ending), ending)
            for ending in action_endings
            if response.find(ending) != -1
        ]
        if not candidates:
            return response
        end_idx, ending = min(candidates, key=lambda item: item[0])
        return response[:end_idx] + ending

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                            next_obs_ids: torch.Tensor) -> Any:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = int(new_attention_mask.sum(dim=1).max().item())
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: Optional[torch.Tensor] = None,
                pad_to_left: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the tool response block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # tool response mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: Optional[torch.Tensor] = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = int(self.tensor_fn.create_attention_mask(responses).sum(dim=1).max().item())
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Any:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        action_reason_stats: DefaultDict[str, int] = defaultdict(int)
        planner_seen = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        self._rollout_debug_samples_remaining = self._rollout_debug_sample_limit()
        self._rollout_debug_samples_emitted = 0

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, reason_stats, action_reasons = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                planner_seen=planner_seen,
                return_reason_stats=True,
                return_reasons=True,
            )
            for reason, count in reason_stats.items():
                action_reason_stats[reason] += count
            plan_only_mask = self._plan_only_mask_from_reasons(action_reasons, active_mask)
            responses_ids_for_state = self._accepted_response_ids_for_state(
                responses_str,
                responses_ids,
                active_mask,
                plan_only_mask,
            )
            
            planner_seen = planner_seen | self._detect_plan_blocks(responses_str, active_mask)
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            next_obs_ids_for_final = self._mask_plan_only_observations_for_final_trajectory(
                next_obs_ids,
                plan_only_mask,
            )
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids_for_state,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids_for_state,
                next_obs_ids_for_final
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, reason_stats, action_reasons = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                do_search=False,
                planner_seen=planner_seen,
                allow_plan_only=False,
                return_reason_stats=True,
                return_reasons=True,
            )
            for reason, count in reason_stats.items():
                action_reason_stats[reason] += count
            plan_only_mask = self._plan_only_mask_from_reasons(action_reasons, active_mask)
            responses_ids_for_state = self._accepted_response_ids_for_state(
                responses_str,
                responses_ids,
                active_mask,
                plan_only_mask,
            )

            planner_seen = planner_seen | self._detect_plan_blocks(responses_str, active_mask)
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids_for_state,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['action_reason_stats'] = dict(action_reason_stats)
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Any:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_data = DataProto.from_dict(final_output)
        final_data.meta_info.update(meta_info)
        
        return final_data

    def _detect_plan_blocks(self, predictions: List[str], active_mask=None) -> torch.Tensor:
        if active_mask is None:
            active_mask = [True] * len(predictions)
        flags = []
        for prediction, active in zip(predictions, active_mask):
            flags.append(self._mask_value_to_bool(active) and self._has_valid_plan(prediction))
        return torch.tensor(flags, dtype=torch.bool)

    def _has_valid_plan(self, prediction: str) -> bool:
        plan_matches = list(re.finditer(r"<plan>(.*?)</plan>", prediction, re.DOTALL))
        if len(plan_matches) != 1:
            return False
        if prediction[:plan_matches[0].start()].strip():
            return False
        if re.search(
            r"</?(?:reasoning|tool_call|tool_response|answer|search|think|information)\b",
            plan_matches[0].group(1),
        ):
            return False
        steps = re.findall(
            r"(?:^|\n)\s*Step\s+\d+\s*:\s*Search\s+(.+?)(?=\n\s*Step\s+\d+\s*:\s*Search\s+|\Z)",
            plan_matches[0].group(1),
            re.DOTALL | re.IGNORECASE,
        )
        return any(step.strip() for step in steps)

    def _extract_accepted_plan_prefix(self, prediction: str) -> str:
        plan_match = re.search(r"<plan>.*?</plan>", prediction, re.DOTALL)
        if not plan_match:
            return prediction
        return prediction[:plan_match.end()]

    def _is_valid_plan_only_turn(self, prediction: str, has_planned: bool) -> bool:
        if has_planned or not self._has_valid_plan(prediction):
            return False
        plan_match = re.search(r"<plan>.*?</plan>", prediction, re.DOTALL)
        if not plan_match:
            return False
        post_plan = prediction[plan_match.end():]
        if re.search(
            r"</?(?:plan|tool_call|tool_response|answer|search|think|information)\b",
            post_plan,
            re.DOTALL,
        ):
            return False
        tag_names = re.findall(r"</?([A-Za-z_][\w-]*)\b[^>]*>", post_plan)
        return all(tag_name == "reasoning" for tag_name in tag_names)

    def _plan_only_mask_from_reasons(self, reasons: List[str], active_mask) -> List[bool]:
        return [
            reason == "valid_plan" and self._mask_value_to_bool(active)
            for reason, active in zip(reasons, active_mask)
        ]

    def _accepted_response_ids_for_state(
        self,
        responses_str: List[str],
        responses_ids: torch.Tensor,
        active_mask,
        plan_only_mask: List[bool],
    ) -> torch.Tensor:
        if not any(plan_only_mask):
            return responses_ids

        accepted_responses = [
            self._extract_accepted_plan_prefix(response) if is_plan_only else response
            for response, is_plan_only in zip(responses_str, plan_only_mask)
        ]
        active_responses = [
            response
            for response, active in zip(accepted_responses, active_mask)
            if self._mask_value_to_bool(active)
        ]
        accepted_response_ids = self._batch_tokenize(active_responses)
        padded_response_ids, _ = self.tensor_fn._example_level_pad(
            accepted_response_ids,
            active_responses,
            active_mask,
        )
        return padded_response_ids

    def _mask_plan_only_observations_for_final_trajectory(
        self,
        next_obs_ids: torch.Tensor,
        plan_only_mask: List[bool],
    ) -> torch.Tensor:
        if not any(plan_only_mask):
            return next_obs_ids

        final_next_obs_ids = next_obs_ids.clone()
        for row_idx, is_plan_only in enumerate(plan_only_mask):
            if is_plan_only:
                final_next_obs_ids[row_idx].fill_(self.tokenizer.pad_token_id)
        return final_next_obs_ids

    def _mask_value_to_bool(self, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.detach().reshape(-1)[0].item())
        return bool(value)

    def _has_single_reasoning_block(self, text: str) -> bool:
        return bool(
            re.fullmatch(r"\s*<reasoning>.*?</reasoning>\s*", text, re.DOTALL)
        )

    def _action_has_required_context(
        self,
        prediction: str,
        match: re.Match,
        has_planned: bool,
    ) -> bool:
        if has_planned:
            if re.search(r"</?plan>", prediction):
                return False
            pre_action_text = prediction[:match.start()]
        else:
            plan_matches = list(re.finditer(r"<plan>(.*?)</plan>", prediction, re.DOTALL))
            if len(plan_matches) != 1:
                return False
            if match.start() < plan_matches[0].end():
                return False
            pre_action_text = prediction[plan_matches[0].end():match.start()]

        return self._has_single_reasoning_block(pre_action_text)

    def _is_valid_search_query(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        if re.search(r"</?[^>]+>", query):
            return False
        if re.search(r"https?://|www\.", query, re.IGNORECASE):
            return False
        if len(query.split()) > 32:
            return False
        return True

    def _rollout_debug_sample_limit(self) -> int:
        raw_limit = os.environ.get("SEARCH_P1_ROLLOUT_DEBUG_SAMPLES", "0")
        try:
            return max(0, int(raw_limit))
        except ValueError:
            print(
                "[SEARCH_P1_ROLLOUT_DEBUG] Ignoring invalid "
                f"SEARCH_P1_ROLLOUT_DEBUG_SAMPLES={raw_limit!r}; expected int."
            )
            return 0

    def _clip_debug_text(self, text: Any, max_chars: int = 240) -> str:
        text = "" if text is None else str(text)
        text = text.replace("\n", "\\n")
        if len(text) > max_chars:
            return text[:max_chars] + "...<truncated>"
        return text

    def _maybe_print_rollout_debug_samples(
        self,
        predictions: List[str],
        next_obs: List[str],
        reasons: List[str],
        active_mask,
        planner_seen,
    ) -> None:
        remaining = getattr(self, "_rollout_debug_samples_remaining", 0)
        if remaining <= 0:
            return
        if planner_seen is None:
            planner_seen = [False] * len(predictions)

        emitted = getattr(self, "_rollout_debug_samples_emitted", 0)
        for prediction, observation, reason, active, has_planned in zip(
            predictions, next_obs, reasons, active_mask, planner_seen
        ):
            if remaining <= 0:
                break
            if not self._mask_value_to_bool(active):
                continue
            has_planned = self._mask_value_to_bool(has_planned)
            print(
                "[SEARCH_P1_ROLLOUT_DEBUG] "
                f"sample={emitted} reason={reason} planner_seen={has_planned} "
                f"prediction={self._clip_debug_text(prediction)} "
                f"observation={self._clip_debug_text(observation)}"
            )
            emitted += 1
            remaining -= 1

        self._rollout_debug_samples_emitted = emitted
        self._rollout_debug_samples_remaining = remaining

    def _missing_plan_reason(self, prediction: str) -> str:
        plan_matches = list(re.finditer(r"<plan>(.*?)</plan>", prediction, re.DOTALL))
        if len(plan_matches) == 0:
            return "missing_plan"
        if len(plan_matches) > 1:
            return "duplicate_plan"
        return "missing_or_invalid_plan_steps"

    def _invalid_action_context_reason(
        self,
        prediction: str,
        match: re.Match,
        has_planned: bool,
    ) -> str:
        if has_planned:
            if re.search(r"</?plan>", prediction):
                return "duplicate_plan"
            pre_action_text = prediction[:match.start()]
        else:
            plan_matches = list(re.finditer(r"<plan>(.*?)</plan>", prediction, re.DOTALL))
            if len(plan_matches) != 1:
                return self._missing_plan_reason(prediction)
            if match.start() < plan_matches[0].end():
                return "action_before_plan"
            pre_action_text = prediction[plan_matches[0].end():match.start()]

        if not re.search(r"<reasoning>.*?</reasoning>", pre_action_text, re.DOTALL):
            if re.search(r"</?(?:search|think|information)\b", pre_action_text, re.DOTALL):
                return "malformed_action_tag"
            return "missing_reasoning"
        return "unknown_invalid"

    def _missing_action_tag_reason(self, prediction: str, has_planned: bool = False) -> str:
        if not prediction or not prediction.strip():
            return "empty_prediction"
        if has_planned and re.search(r"</?plan>", prediction):
            return "duplicate_plan"
        if re.search(
            r"</?(tool_call|answer|search|think|information)\b[^>]*>",
            prediction,
            re.DOTALL,
        ):
            return "malformed_action_tag"
        return "missing_action_tag"

    def _invalid_action_observation(self, has_planned: bool, reason: str) -> str:
        if has_planned:
            base = (
                "My previous action is invalid. A valid <plan> has already been accepted. "
                "Do not output <plan> again. Output exactly one "
                "<reasoning>...</reasoning> block followed by exactly one "
                "<tool_call>...</tool_call> or <answer>...</answer> block."
            )
        else:
            base = (
                "My previous action is invalid. No valid <plan> has been accepted yet. "
                "First output one complete <plan>...</plan> with numbered Search steps, "
                "then output one <reasoning>...</reasoning> block followed by one "
                "<tool_call>...</tool_call> or <answer>...</answer> block."
            )

        extra = ""
        if reason == "malformed_action_tag":
            extra = (
                " Old <search>, <think>, and <information> tags are deprecated; "
                "the trajectory vocabulary is <reasoning>, <tool_call>, and "
                "<tool_response>."
            )
        elif reason == "missing_action_tag":
            extra = (
                " This turn must end with a legal action tag: "
                "</tool_call> for search or </answer> for the final answer."
            )
        elif reason == "missing_reasoning":
            extra = " Reasoning must immediately precede the action tag."
        elif reason == "duplicate_plan":
            if has_planned:
                extra = " Repeating <plan> after a valid plan is invalid."
            else:
                extra = " Output exactly one valid <plan> block."

        return f"\n{base}{extra} Let me try again.\n"

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
        planner_seen=None,
        return_reason_stats=False,
        allow_plan_only=True,
        return_reasons=False,
    ) -> Tuple:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            Tuple of observation strings, done flags, valid-action flags, search flags,
            and optional reason stats when return_reason_stats is True.
        """
        if active_mask is None:
            active_mask = [True] * len(predictions)
        cur_actions, contents, reasons = self.postprocess_predictions(
            predictions,
            planner_seen=planner_seen,
            active_mask=active_mask,
            allow_plan_only=allow_plan_only,
            return_reasons=True,
        )
        reason_stats: DefaultDict[str, int] = defaultdict(int)
        for reason in reasons:
            reason_stats[reason] += 1
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        search_results: List[str]
        if do_search and search_queries:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        if planner_seen is None:
            planner_seen = [False] * len(predictions)

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            active = self._mask_value_to_bool(active)
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'plan':
                    next_obs.append(self.PLAN_ACCEPTED_OBSERVATION)
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<tool_response>{search_results.pop(0).strip()}</tool_response>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    has_planned_for_feedback = self._mask_value_to_bool(
                        planner_seen[i]
                    ) or self._has_valid_plan(predictions[i])
                    next_obs.append(
                        self._invalid_action_observation(
                            has_planned_for_feedback,
                            reasons[i],
                        )
                    )
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0
        self._maybe_print_rollout_debug_samples(
            predictions,
            next_obs,
            reasons,
            active_mask,
            planner_seen,
        )

        if return_reason_stats and return_reasons:
            return next_obs, dones, valid_action, is_search, dict(reason_stats), reasons
        if return_reason_stats:
            return next_obs, dones, valid_action, is_search, dict(reason_stats)
        if return_reasons:
            return next_obs, dones, valid_action, is_search, reasons
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(
        self,
        predictions: List[Any],
        planner_seen=None,
        active_mask=None,
        allow_plan_only=True,
        return_reasons=False,
    ) -> Tuple:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, contents list), with reasons appended when
            return_reasons is True.
        """
        actions: List[Any] = []
        contents: List[str] = []
        reasons: List[str] = []
                
        if planner_seen is None:
            planner_seen = [False] * len(predictions)
        if active_mask is None:
            active_mask = [True] * len(predictions)

        for prediction, has_planned, active in zip(predictions, planner_seen, active_mask):
            has_planned = self._mask_value_to_bool(has_planned)
            active = self._mask_value_to_bool(active)
            if isinstance(prediction, str): # for llm output
                reason = "inactive"
                if not active:
                    actions.append(None)
                    contents.append('')
                    reasons.append(reason)
                    continue
                pattern = r'<(tool_call|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action_tag = match.group(1)
                    action = 'search' if action_tag == 'tool_call' else action_tag
                    has_plan = has_planned or self._has_valid_plan(prediction)
                    if active and not has_plan:
                        reason = self._missing_plan_reason(prediction)
                        content = ''
                        action = None
                    elif active and not self._action_has_required_context(prediction, match, has_planned):
                        reason = self._invalid_action_context_reason(prediction, match, has_planned)
                        content = ''
                        action = None
                    elif action == 'search' and not self._is_valid_search_query(content):
                        if active:
                            reason = "invalid_tool_call"
                        content = ''
                        action = None
                    elif active:
                        reason = f"valid_{action}"
                else:
                    content = ''
                    action = None
                    if active:
                        if allow_plan_only and self._is_valid_plan_only_turn(prediction, has_planned):
                            action = 'plan'
                            reason = "valid_plan"
                        else:
                            reason = self._missing_action_tag_reason(prediction, has_planned)
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            reasons.append(reason)
            
        if return_reasons:
            return actions, contents, reasons
        return actions, contents

    def batch_search(self, queries: Optional[List[str]] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
