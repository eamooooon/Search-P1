import re

from search_p1.analysis.reference_io import (
    key_metadata,
    read_jsonl,
    row_ground_truth,
    row_key,
    solution_str,
)
from search_p1.analysis.reward_format import qa_em_format


def extract_final_answer(text):
    answer = qa_em_format.extract_solution(text)
    if answer is not None:
        return answer
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def is_correct_trajectory(row):
    text = solution_str(row)
    if not text:
        return False
    ground_truth = row_ground_truth(row)
    targets = ground_truth.get("target")
    if not targets:
        return False
    answer = extract_final_answer(text)
    if answer is None:
        return False
    return bool(qa_em_format.em_check(answer, targets))


def valid_actions(row):
    actions = qa_em_format.extract_search_calls(solution_str(row))
    return [action for action in actions if qa_em_format.is_valid_search_query(action)]


def collect_successful_groups(path, max_successful_per_question=None):
    groups = {}
    total_rows = 0
    correct_rows = 0
    skipped_no_key = 0
    skipped_no_actions = 0

    for row in read_jsonl(path):
        total_rows += 1
        key = row_key(row)
        if key is None:
            skipped_no_key += 1
            continue
        group = groups.setdefault(key, {
            "metadata": key_metadata(row),
            "total": 0,
            "correct": 0,
            "trajectories": [],
        })
        group["total"] += 1

        if not is_correct_trajectory(row):
            continue
        correct_rows += 1
        actions = valid_actions(row)
        if not actions:
            skipped_no_actions += 1
            continue
        if max_successful_per_question is None or len(group["trajectories"]) < max_successful_per_question:
            group["trajectories"].append({
                "actions": actions,
                "solution_str": solution_str(row),
            })
        group["correct"] += 1

    stats = {
        "total_rows": total_rows,
        "correct_rows": correct_rows,
        "correct_rows_with_actions": correct_rows - skipped_no_actions,
        "groups": len(groups),
        "groups_with_correct": sum(1 for group in groups.values() if group["correct"] > 0),
        "groups_with_correct_actions": sum(1 for group in groups.values() if group["trajectories"]),
        "skipped_no_key": skipped_no_key,
        "skipped_correct_no_actions": skipped_no_actions,
    }
    return groups, stats
