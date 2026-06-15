import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from search_p1.analysis.analyze_reference_rollouts import (
    contains_match,
    gold_in_information,
    relaxed_answer_match,
    targets_from_row,
    token_f1,
)
from search_p1.analysis.reference_io import read_jsonl, row_key
from search_p1.analysis.reference_sampling import extract_final_answer, valid_actions
from search_p1.analysis.reward_format import qa_em_format


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def final_answer(row):
    answer = row.get("final_answer")
    if answer is not None:
        return answer
    return extract_final_answer(row.get("trajectory") or row.get("solution_str") or "")


def best_target(answer, targets, relaxed_f1_threshold):
    best = None
    best_score = -1.0
    for target in targets:
        f1 = token_f1(answer, target)
        contains = contains_match(answer, target)
        relaxed = contains or f1 >= relaxed_f1_threshold
        score = (1.0 if relaxed else 0.0) + f1 + (0.1 if contains else 0.0)
        if score > best_score:
            best = target
            best_score = score
    return best


def information_target(row, targets):
    for target in targets:
        if gold_in_information(row, [target]):
            return target
    return None


def replace_final_answer(text, corrected_answer):
    matches = list(ANSWER_PATTERN.finditer(text or ""))
    if not matches:
        return text
    match = matches[-1]
    return text[:match.start()] + f"<answer>{corrected_answer}</answer>" + text[match.end():]


def classify_usable(row, relaxed_f1_threshold):
    targets = targets_from_row(row)
    answer = final_answer(row)
    if not targets or not answer:
        return None

    strict = bool(qa_em_format.em_check(answer, targets))
    if strict:
        corrected = best_target(answer, targets, relaxed_f1_threshold) or targets[0]
        return {
            "category": "strict_correct",
            "original_answer": answer,
            "corrected_answer": corrected,
            "needs_correction": False,
        }

    if relaxed_answer_match(answer, targets, relaxed_f1_threshold):
        corrected = best_target(answer, targets, relaxed_f1_threshold) or targets[0]
        return {
            "category": "relaxed_answer_match",
            "original_answer": answer,
            "corrected_answer": corrected,
            "needs_correction": True,
        }

    corrected = information_target(row, targets)
    if corrected:
        return {
            "category": "gold_in_information_wrong_answer",
            "original_answer": answer,
            "corrected_answer": corrected,
            "needs_correction": True,
        }

    return None


def compact_corrected_row(row, usable):
    solution = row.get("solution_str") or row.get("trajectory") or ""
    trajectory = row.get("trajectory") or solution
    corrected_solution = replace_final_answer(solution, usable["corrected_answer"])
    corrected_trajectory = replace_final_answer(trajectory, usable["corrected_answer"])
    reward_components = row.get("reward_components") or {}

    return {
        "schema_version": 1,
        "data_source": row.get("data_source"),
        "split": row.get("split"),
        "index": row.get("index"),
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "solution_str": corrected_solution,
        "trajectory": corrected_trajectory,
        "search_calls": row.get("search_calls") or valid_actions(row),
        "plan_steps": row.get("plan_steps") or qa_em_format.extract_plan_steps(trajectory),
        "final_answer": usable["corrected_answer"],
        "rollout_index": row.get("rollout_index"),
        "trajectory_index": row.get("trajectory_index"),
        "collection": {
            "accepted_success": True,
            "corrected_reference": True,
            "category": usable["category"],
            "needs_correction": usable["needs_correction"],
            "original_answer": usable["original_answer"],
            "corrected_answer": usable["corrected_answer"],
            "original_final_score": reward_components.get("final_score"),
            "original_base_score": reward_components.get("base_score"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export all usable reference rollout trajectories, correcting relaxed/evidence-gold "
            "final answers to gold answers."
        )
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_usable_per_question", type=int, default=2)
    parser.add_argument("--relaxed_f1_threshold", type=float, default=0.6)
    args = parser.parse_args()

    grouped_rows = defaultdict(list)
    stats = Counter()
    category_counts = Counter()

    for path in args.jsonl:
        for row in read_jsonl(path):
            stats["input_rows"] += 1
            key = row_key(row)
            if key is None:
                stats["skipped_no_key"] += 1
                continue
            usable = classify_usable(row, args.relaxed_f1_threshold)
            if usable is None:
                stats["not_usable_rows"] += 1
                continue
            actions = row.get("search_calls") or valid_actions(row)
            if not actions:
                stats["skipped_no_valid_actions"] += 1
                continue
            category_counts[usable["category"]] += 1
            grouped_rows[key].append(compact_corrected_row(row, usable))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    emitted_questions = 0
    emitted_rows = 0
    eligible_dist = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for key, rows in grouped_rows.items():
            usable_count = len(rows)
            eligible_dist[usable_count] += 1
            if usable_count < args.min_usable_per_question:
                stats["skipped_questions_below_min"] += 1
                stats["skipped_rows_below_min"] += usable_count
                continue
            emitted_questions += 1
            for row in rows:
                emitted_rows += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "inputs": [str(path) for path in args.jsonl],
        "output": str(args.output),
        "min_usable_per_question": args.min_usable_per_question,
        "input_rows": stats["input_rows"],
        "usable_questions_before_min_filter": len(grouped_rows),
        "usable_rows_before_min_filter": sum(len(rows) for rows in grouped_rows.values()),
        "emitted_questions": emitted_questions,
        "emitted_rows": emitted_rows,
        "category_counts_before_min_filter": dict(category_counts),
        "usable_count_per_question_dist": [
            {"usable_count": count, "questions": questions}
            for count, questions in eligible_dist.most_common()
        ],
        "skipped_no_key": stats["skipped_no_key"],
        "skipped_no_valid_actions": stats["skipped_no_valid_actions"],
        "not_usable_rows": stats["not_usable_rows"],
        "skipped_questions_below_min": stats["skipped_questions_below_min"],
        "skipped_rows_below_min": stats["skipped_rows_below_min"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
