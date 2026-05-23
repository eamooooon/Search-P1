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

import re
import string
import logging

logger = logging.getLogger(__name__)

_STEP_LINE_PATTERN = re.compile(
    r"^\s*Step\s+(\d+)\s*:\s*Search\s+(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TAG_PATTERN = re.compile(r"</?[^>]+>")
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "by",
    "find",
    "for",
    "in",
    "needed",
    "of",
    "on",
    "query",
    "question",
    "relevant",
    "search",
    "specific",
    "the",
    "to",
}
_INTENT_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]")
_INTENT_STOPWORDS = {
    "actor",
    "actress",
    "details",
    "determine",
    "entity",
    "film",
    "find",
    "identified",
    "identify",
    "information",
    "item",
    "movie",
    "person",
    "relevant",
    "result",
    "specific",
    "target",
    "thing",
}
_INTENT_GENERIC_SINGLE_OVERLAP = {
    "age",
    "birth",
    "birthplace",
    "character",
    "city",
    "country",
    "date",
    "location",
    "name",
    "nationality",
    "place",
    "role",
    "title",
    "year",
}
_SUPPORTED_PATH_MATCH_STRATEGIES = {"intent_lexical", "lexical"}


def validate_path_match_strategy(match_strategy):
    if match_strategy not in _SUPPORTED_PATH_MATCH_STRATEGIES:
        supported = ", ".join(sorted(_SUPPORTED_PATH_MATCH_STRATEGIES))
        raise ValueError(
            f"Unsupported path_match_strategy '{match_strategy}'. "
            f"Supported strategies: {supported}. "
            "Embedding and offline LLM matching are not implemented."
        )


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _extract_assistant_content(text):
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    if assistant_match:
        return text[assistant_match.end():]
    return text


def extract_plan_steps(text):
    content = _extract_assistant_content(text)
    match = re.search(r"<plan>(.*?)</plan>", content, re.DOTALL)
    if not match:
        return []
    plan_text = match.group(1)
    steps = [match.group(2).strip() for match in _STEP_LINE_PATTERN.finditer(plan_text)]
    return [step.strip() for step in steps if step.strip()]


def extract_tool_calls(text):
    content = _extract_assistant_content(text)
    content = re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    return [match.strip() for match in matches]


def extract_search_queries(text):
    return extract_tool_calls(text)


def count_actions(text):
    return len(extract_tool_calls(text))


def has_legal_tool_call(text):
    return any(is_valid_search_query(action) for action in extract_tool_calls(text))


def validate_planner_steps(steps):
    return bool(steps) and all(step and not _TAG_PATTERN.search(step) for step in steps)


def _extract_plan_text(text):
    content = _extract_assistant_content(text)
    match = re.search(r"<plan>(.*?)</plan>", content, re.DOTALL)
    return match.group(1) if match else ""


def _has_single_front_loaded_plan(text):
    content = _extract_assistant_content(text)
    if len(re.findall(r"<plan>", content)) != 1 or len(re.findall(r"</plan>", content)) != 1:
        return False
    return bool(re.match(r"^\s*<plan>.*?</plan>", content, re.DOTALL))


def validate_planner_block(text, steps=None, max_plan_steps=None):
    if steps is None:
        steps = extract_plan_steps(text)
    if not _has_single_front_loaded_plan(text) or not validate_planner_steps(steps):
        return False
    if max_plan_steps is not None and len(steps) > max_plan_steps:
        return False

    plan_text = _extract_plan_text(text)
    nonempty_lines = [line for line in plan_text.splitlines() if line.strip()]
    step_matches = list(_STEP_LINE_PATTERN.finditer(plan_text))
    if len(nonempty_lines) != len(step_matches):
        return False

    step_numbers = [int(match.group(1)) for match in step_matches]
    return step_numbers == list(range(1, len(step_numbers) + 1))


def validate_actions(actions):
    return bool(actions) and all(is_valid_search_query(action) for action in actions)


def normalize_step(text):
    normalized = normalize_answer(text)
    tokens = [token for token in normalized.split() if token not in _MATCH_STOPWORDS]
    return " ".join(tokens)


def normalize_intent_step(text):
    text = _INTENT_PLACEHOLDER_PATTERN.sub(" ", text)
    normalized = normalize_step(text)
    tokens = [token for token in normalized.split() if token not in _INTENT_STOPWORDS]
    return " ".join(tokens)


def _lexical_step_matches_action(step_text, action_text):
    if not step_text or not action_text:
        return False
    if step_text in action_text or action_text in step_text:
        return True

    step_tokens = set(step_text.split())
    action_tokens = set(action_text.split())
    if not step_tokens or not action_tokens:
        return False

    overlap = step_tokens & action_tokens
    shorter_len = min(len(step_tokens), len(action_tokens))
    if shorter_len == 1:
        return bool(overlap) and len(step_tokens | action_tokens) <= 3
    return len(overlap) / shorter_len >= 0.5


def _intent_lexical_step_matches_action(step, action):
    step_text = normalize_step(step)
    action_text = normalize_step(action)
    if _lexical_step_matches_action(step_text, action_text):
        return True

    intent_text = normalize_intent_step(step)
    intent_tokens = set(intent_text.split())
    action_tokens = set(action_text.split())
    if len(intent_tokens) < 2 or not action_tokens:
        return False

    overlap = intent_tokens & action_tokens
    if not overlap:
        return False
    if len(overlap) == 1 and next(iter(overlap)) in _INTENT_GENERIC_SINGLE_OVERLAP:
        return False
    return len(overlap) >= 2 or (len(overlap) / len(intent_tokens)) >= 0.5


def step_matches_action(step, action, match_strategy="lexical"):
    validate_path_match_strategy(match_strategy)
    step_text = normalize_step(step)
    action_text = normalize_step(action)
    if match_strategy == "lexical":
        return _lexical_step_matches_action(step_text, action_text)
    return _intent_lexical_step_matches_action(step, action)


def count_covered_steps(steps, actions, match_strategy="lexical"):
    validate_path_match_strategy(match_strategy)
    unique_actions = []
    seen_actions = set()
    for action in actions:
        normalized_action = normalize_step(action)
        if not normalized_action or normalized_action in seen_actions:
            continue
        seen_actions.add(normalized_action)
        unique_actions.append(action)

    matched_actions = set()
    covered = 0
    for step in steps:
        for action_index, action in enumerate(unique_actions):
            if action_index in matched_actions:
                continue
            if step_matches_action(step, action, match_strategy=match_strategy):
                matched_actions.add(action_index)
                covered += 1
                break
    return covered


def compute_self_consistency_score(solution_str, match_strategy="lexical", max_plan_steps=None):
    return compute_self_consistency_components(
        solution_str,
        match_strategy=match_strategy,
        max_plan_steps=max_plan_steps,
    )["self_consistency"]


def compute_self_consistency_components(solution_str, match_strategy="lexical", max_plan_steps=None):
    validate_path_match_strategy(match_strategy)
    steps = extract_plan_steps(solution_str)
    actions = extract_tool_calls(solution_str)
    r_planner = 1.0 if validate_planner_block(solution_str, steps, max_plan_steps=max_plan_steps) else 0.0
    n_plan = len(steps)
    n_actions = len(actions)
    n_exec_self = 0

    if r_planner != 0 and n_plan > 0 and n_actions > 0 and validate_actions(actions):
        n_exec_self = count_covered_steps(steps, actions, match_strategy=match_strategy)

    self_consistency = 0.0
    if r_planner != 0 and n_plan > 0 and n_actions > 0:
        self_consistency = r_planner * (n_exec_self / n_plan) * (n_exec_self / n_actions)

    return {
        "self_consistency": self_consistency,
        "self_r_planner": r_planner,
        "self_n_plan": n_plan,
        "self_n_actions": n_actions,
        "self_n_exec": n_exec_self,
    }


def is_valid_search_query(query):
    if not query or not query.strip():
        return False
    if re.search(r"</?[^>]+>", query):
        return False
    if _URL_PATTERN.search(query):
        return False
    if len(query.split()) > 32:
        return False
    return True


def is_valid_sequence(text, max_plan_steps=None):
    content = _extract_assistant_content(text)

    tags_to_check = ["plan", "reasoning", "tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    plan_count = len(re.findall(r"<plan>", content))
    if plan_count > 1:
        return False, "Multiple <plan> blocks are not allowed"
    if plan_count != 1:
        return False, "Missing required <plan> block"
    if not validate_planner_block(content, max_plan_steps=max_plan_steps):
        return False, "Missing or invalid plan steps"

    split_pattern = r"(</?(?:plan|reasoning|tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    current_tool_call = ""

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:plan|reasoning|tool_call|tool_response|answer)>", part):
            if part == "<plan>" and state == "start":
                state = "in_plan"
            elif part == "</plan>" and state == "in_plan":
                state = "after_plan"
            elif part == "<reasoning>" and state in ["after_plan", "tool_response"]:
                state = "in_reasoning"
            elif part == "</reasoning>" and state == "in_reasoning":
                state = "after_reasoning"
            elif part == "<tool_call>" and state == "after_reasoning":
                state = "in_tool_call"
                current_tool_call = ""
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
                if not is_valid_search_query(current_tool_call):
                    return False, "Invalid tool call"
            elif part == "<tool_response>" and state == "after_tool_call":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "tool_response"
            elif part == "<answer>" and state == "after_reasoning":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_plan", "in_reasoning", "in_tool_call", "in_tool_response", "in_answer"]:
                if state == "in_tool_call":
                    current_tool_call += part
                pass
            elif state in ["start", "after_plan", "after_reasoning", "after_tool_call", "tool_response", "end"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    assistant_marker = "<|im_start|>assistant"
    has_assistant_marker = assistant_marker in solution_str
    if has_assistant_marker:
        solution_str = solution_str.rsplit(assistant_marker, 1)[1]
        solution_str = solution_str.split("<|im_end|>", 1)[0]
    solution_str = re.sub(
        r"My previous action is invalid\.[^\n]*Let me try again\.",
        "",
        solution_str,
    )

    answer_pattern = r'<answer>(.*?)</answer>'
    content = _extract_assistant_content(solution_str)
    match = re.finditer(answer_pattern, content, re.DOTALL)
    matches = list(match)

    if not matches:
        return None

    if not has_assistant_marker and content == solution_str and len(matches) <= 1:
        return None
    
    # Use the final answer tag when a trajectory contains multiple turns.
    return matches[-1].group(1).strip()


def extract_tool_response_blocks(text: str) -> list[str]:
    content = _extract_assistant_content(text)
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, content, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    seqs = extract_tool_response_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str,
                     ground_truth,
                     method='strict',
                     structure_format_score=0,
                     final_format_score=0,
                     retrieval_score=0,
                     format_score=0,
                     score=1.,
                     path_match_strategy="lexical",
                     require_search_for_format=False,
                     max_plan_steps=None,
                     self_consistency_weight=0.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    return compute_score_components(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        structure_format_score=structure_format_score,
        final_format_score=final_format_score,
        retrieval_score=retrieval_score,
        format_score=format_score,
        score=score,
        path_match_strategy=path_match_strategy,
        require_search_for_format=require_search_for_format,
        max_plan_steps=max_plan_steps,
        self_consistency_weight=self_consistency_weight,
    )["final_score"]


def compute_score_components(solution_str,
                             ground_truth,
                             method='strict',
                             structure_format_score=0,
                             final_format_score=0,
                             retrieval_score=0,
                             format_score=0,
                             score=1.,
                             path_match_strategy="lexical",
                             require_search_for_format=False,
                             max_plan_steps=None,
                             self_consistency_weight=0.0):
    validate_path_match_strategy(path_match_strategy)
    is_valid_format, _ = is_valid_sequence(solution_str, max_plan_steps=max_plan_steps)
    has_search = has_legal_tool_call(solution_str)
    format_shaping_allowed = (not require_search_for_format) or has_search
    final_format_shaping_allowed = format_shaping_allowed and (
        (not require_search_for_format) or is_valid_format
    )
    # These component metrics describe whether the search gate allows shaping.
    # They are not a substitute for is_valid_format.
    effective_structure_format = 1.0 if format_shaping_allowed else 0.0
    effective_retrieval = 1.0 if format_shaping_allowed else 0.0
    retrieval_correct = False
    if is_valid_format and format_shaping_allowed:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    logger.debug(
        "Reward score inputs: golden_answers=%s extracted_answer=%s solution=%s",
        ground_truth['target'],
        answer,
        solution_str,
    )
            
    if answer is None:
        if is_valid_format and format_shaping_allowed:
            if retrieval_correct:
                base_score = structure_format_score + retrieval_score # 0.3
            else:
                base_score = structure_format_score # 0.2
        else:
            base_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                base_score = score # 1
            else:
                base_score = score - structure_format_score # 0.8
        elif is_valid_format and format_shaping_allowed:
            if retrieval_correct:
                base_score = structure_format_score + retrieval_score # 0.3
            else:
                base_score = structure_format_score # 0.2
        elif final_format_shaping_allowed:
            base_score = final_format_score # 0.1
        else:
            base_score = 0

    self_components = compute_self_consistency_components(
        solution_str,
        match_strategy=path_match_strategy,
        max_plan_steps=max_plan_steps,
    )
    track_a_bonus = self_consistency_weight * self_components["self_consistency"]
    final_score = base_score + track_a_bonus

    return {
        "base_score": base_score,
        "has_search": has_search,
        "effective_structure_format": effective_structure_format,
        "effective_retrieval": effective_retrieval,
        "track_a_bonus": track_a_bonus,
        "self_consistency_weight": self_consistency_weight,
        "final_score": final_score,
        **self_components,
    }
