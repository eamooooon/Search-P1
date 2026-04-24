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
import random

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
    step_pattern = r"(?:^|\n)\s*Step\s+\d+\s*:\s*Search\s+(.+?)(?=\n\s*Step\s+\d+\s*:\s*Search\s+|\Z)"
    steps = re.findall(step_pattern, plan_text, re.DOTALL | re.IGNORECASE)
    return [step.strip() for step in steps if step.strip()]


def extract_search_queries(text):
    content = _extract_assistant_content(text)
    matches = re.findall(r"<search>(.*?)</search>", content, re.DOTALL)
    return [match.strip() for match in matches]


def is_valid_search_query(query):
    if not query or not query.strip():
        return False
    if re.search(r"</?[^>]+>", query):
        return False
    if re.search(r"https?://|www\.", query, re.IGNORECASE):
        return False
    if len(query.split()) > 32:
        return False
    return True


def is_valid_sequence(text):
    content = _extract_assistant_content(text)

    tags_to_check = ["plan", "think", "search", "information", "answer"]
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
    if not extract_plan_steps(content):
        return False, "Missing valid plan steps"

    split_pattern = r"(</?(?:plan|think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    current_search_query = ""

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:plan|think|search|information|answer)>", part):
            if part == "<plan>" and state == "start":
                state = "in_plan"
            elif part == "</plan>" and state == "in_plan":
                state = "after_plan"
            elif part == "<think>" and state in ["after_plan", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
                current_search_query = ""
            elif part == "</search>" and state == "in_search":
                state = "after_search"
                if not is_valid_search_query(current_search_query):
                    return False, "Invalid search query"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_plan", "in_think", "in_search", "in_information", "in_answer"]:
                if state == "in_search":
                    current_search_query += part
                pass
            elif state in ["start", "after_plan", "after_think", "after_search", "information", "end"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # The decoded sequence contains both prompt and response, and the prompt
    # includes an <answer> example. Require at least two <answer> blocks so
    # that we extract the model's actual final answer instead of the example.
    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
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
                     score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1
