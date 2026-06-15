import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from search_p1.analysis.reference_io import read_jsonl, row_ground_truth, row_key
from search_p1.analysis.reference_sampling import extract_final_answer
from search_p1.analysis.reward_format import qa_em_format


INFORMATION_PATTERN = re.compile(r"<information>(.*?)</information>", re.DOTALL | re.IGNORECASE)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum((pred_counts & target_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction, target):
    pred = normalize_text(prediction)
    gold = normalize_text(target)
    return bool(pred and gold and (pred in gold or gold in pred))


def targets_from_row(row):
    ground_truth = row_ground_truth(row)
    target = ground_truth.get("target") if isinstance(ground_truth, dict) else None
    if isinstance(target, list):
        return [str(item) for item in target if str(item).strip()]
    if target is None:
        return []
    return [str(target)]


def final_answer(row):
    answer = row.get("final_answer")
    if answer is not None:
        return answer
    return extract_final_answer(row.get("trajectory") or row.get("solution_str") or "")


def relaxed_answer_match(answer, targets, threshold):
    if not answer or not targets:
        return False
    return any(contains_match(answer, target) or token_f1(answer, target) >= threshold for target in targets)


def information_text(row):
    trajectory = row.get("trajectory") or row.get("solution_str") or ""
    return "\n".join(match.group(1) for match in INFORMATION_PATTERN.finditer(trajectory))


def gold_in_information(row, targets):
    info = normalize_text(information_text(row))
    return any(normalize_text(target) and normalize_text(target) in info for target in targets)


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def serial_counter(counter, limit=None):
    items = counter.most_common(limit)
    return [{"key": str(key), "count": count} for key, count in items]


def summarize_question_counts(question_counts, key, thresholds):
    return {
        f">={threshold}": sum(1 for counts in question_counts.values() if counts[key] >= threshold)
        for threshold in thresholds
    }


def analyze(paths, relaxed_f1_threshold):
    global_stats = Counter()
    file_stats = {}
    file_question_counts = {}
    question_counts = defaultdict(Counter)
    score_dist = Counter()
    base_score_dist = Counter()
    search_count_dist = Counter()
    plan_count_dist = Counter()
    examples = {
        "score1_correct": [],
        "relaxed_answer_match": [],
        "gold_in_information_wrong_answer": [],
        "no_answer": [],
    }

    for path in paths:
        path = str(path)
        stats = Counter()
        local_question_counts = defaultdict(Counter)
        for row in read_jsonl(path):
            stats["rows"] += 1
            global_stats["rows"] += 1

            key = row_key(row)
            if key is None:
                stats["skipped_no_key"] += 1
                global_stats["skipped_no_key"] += 1
                continue

            q_counts = question_counts[key]
            if q_counts["total"] == 0:
                global_stats["questions_seen"] += 1
            q_counts["total"] += 1
            local_counts = local_question_counts[key]
            if local_counts["total"] == 0:
                stats["questions_seen"] += 1
            local_counts["total"] += 1

            reward_components = row.get("reward_components") or {}
            final_score = reward_components.get("final_score")
            base_score = reward_components.get("base_score")
            score_dist[final_score] += 1
            base_score_dist[base_score] += 1

            answer = final_answer(row)
            targets = targets_from_row(row)
            strict = bool(answer and targets and qa_em_format.em_check(answer, targets))
            relaxed = relaxed_answer_match(answer, targets, relaxed_f1_threshold)
            has_info_gold = (not strict) and gold_in_information(row, targets)
            score1 = final_score == 1.0
            duplicate_plan = bool(float(reward_components.get("duplicate_plan", 0.0) or 0.0))
            has_search = bool(row.get("search_calls"))
            no_answer = not bool(answer)

            search_count = len(row.get("search_calls") or [])
            plan_count = len(row.get("plan_steps") or [])
            search_count_dist[search_count] += 1
            plan_count_dist[plan_count] += 1

            for bucket in (stats, global_stats):
                bucket["strict_em_rows"] += int(strict)
                bucket["score1_rows"] += int(score1)
                bucket["relaxed_answer_rows"] += int((not strict) and relaxed)
                bucket["gold_in_information_wrong_rows"] += int(has_info_gold)
                bucket["has_search_rows"] += int(has_search)
                bucket["no_answer_rows"] += int(no_answer)
                bucket["duplicate_plan_rows"] += int(duplicate_plan)

            q_counts["strict_em"] += int(strict)
            q_counts["score1"] += int(score1)
            q_counts["strict_or_relaxed"] += int(strict or relaxed)
            q_counts["strict_relaxed_or_info_gold"] += int(strict or relaxed or has_info_gold)
            local_counts["strict_em"] += int(strict)
            local_counts["score1"] += int(score1)
            local_counts["strict_or_relaxed"] += int(strict or relaxed)
            local_counts["strict_relaxed_or_info_gold"] += int(strict or relaxed or has_info_gold)

            if score1 and len(examples["score1_correct"]) < 5:
                examples["score1_correct"].append(example_row(row, answer, targets, final_score))
            if (not strict) and relaxed and len(examples["relaxed_answer_match"]) < 5:
                examples["relaxed_answer_match"].append(example_row(row, answer, targets, final_score))
            if has_info_gold and len(examples["gold_in_information_wrong_answer"]) < 5:
                examples["gold_in_information_wrong_answer"].append(example_row(row, answer, targets, final_score))
            if no_answer and len(examples["no_answer"]) < 5:
                examples["no_answer"].append(example_row(row, answer, targets, final_score))

        file_stats[path] = stats
        file_question_counts[path] = local_question_counts

    question_total = len(question_counts)
    strict_keys = {key for key, counts in question_counts.items() if counts["strict_em"] > 0}
    relaxed_keys = {key for key, counts in question_counts.items() if counts["strict_or_relaxed"] > 0}
    info_gold_keys = {
        key for key, counts in question_counts.items()
        if counts["strict_relaxed_or_info_gold"] > 0
    }

    report = {
        "inputs": [str(path) for path in paths],
        "rows": global_stats["rows"],
        "questions": question_total,
        "overall": rates(global_stats),
        "per_file": {path: rates(stats) for path, stats in file_stats.items()},
        "per_file_question_coverage": {
            path: question_coverage(counts)
            for path, counts in file_question_counts.items()
        },
        "question_coverage": {
            "strict_em": coverage(question_counts, "strict_em", question_total),
            "score1": coverage(question_counts, "score1", question_total),
            "strict_or_relaxed_answer": coverage(question_counts, "strict_or_relaxed", question_total),
            "strict_relaxed_or_gold_in_information": coverage(
                question_counts,
                "strict_relaxed_or_info_gold",
                question_total,
            ),
            "new_questions_from_relaxed_answer": len(relaxed_keys - strict_keys),
            "new_questions_from_gold_in_information": len(info_gold_keys - relaxed_keys),
            "strict_em_thresholds": summarize_question_counts(question_counts, "strict_em", [1, 3, 5, 8, 16]),
            "score1_thresholds": summarize_question_counts(question_counts, "score1", [1, 3, 5, 8, 16]),
        },
        "distributions": {
            "final_score": serial_counter(score_dist),
            "base_score": serial_counter(base_score_dist),
            "search_count": serial_counter(search_count_dist),
            "plan_count": serial_counter(plan_count_dist),
        },
        "examples": examples,
    }
    return report


def rates(stats):
    rows = stats["rows"]
    questions = stats.get("questions_seen", 0)
    return {
        "rows": rows,
        "questions_seen": questions,
        "strict_em_rows": stats["strict_em_rows"],
        "strict_em_rate": pct(stats["strict_em_rows"], rows),
        "score1_rows": stats["score1_rows"],
        "score1_rate": pct(stats["score1_rows"], rows),
        "relaxed_answer_rows": stats["relaxed_answer_rows"],
        "relaxed_answer_rate": pct(stats["relaxed_answer_rows"], rows),
        "gold_in_information_wrong_rows": stats["gold_in_information_wrong_rows"],
        "gold_in_information_wrong_rate": pct(stats["gold_in_information_wrong_rows"], rows),
        "has_search_rows": stats["has_search_rows"],
        "has_search_rate": pct(stats["has_search_rows"], rows),
        "no_answer_rows": stats["no_answer_rows"],
        "no_answer_rate": pct(stats["no_answer_rows"], rows),
        "duplicate_plan_rows": stats["duplicate_plan_rows"],
        "duplicate_plan_rate": pct(stats["duplicate_plan_rows"], rows),
        "skipped_no_key": stats["skipped_no_key"],
    }


def question_coverage(question_counts):
    question_total = len(question_counts)
    return {
        "questions": question_total,
        "strict_em": coverage(question_counts, "strict_em", question_total),
        "score1": coverage(question_counts, "score1", question_total),
        "strict_or_relaxed_answer": coverage(question_counts, "strict_or_relaxed", question_total),
        "strict_relaxed_or_gold_in_information": coverage(
            question_counts,
            "strict_relaxed_or_info_gold",
            question_total,
        ),
        "strict_em_thresholds": summarize_question_counts(question_counts, "strict_em", [1, 3, 5, 8, 16]),
        "score1_thresholds": summarize_question_counts(question_counts, "score1", [1, 3, 5, 8, 16]),
    }


def coverage(question_counts, key, question_total):
    covered = sum(1 for counts in question_counts.values() if counts[key] > 0)
    return {
        "questions": covered,
        "rate": pct(covered, question_total),
    }


def example_row(row, answer, targets, final_score):
    return {
        "index": row.get("index"),
        "rollout_index": row.get("rollout_index"),
        "question": row.get("question"),
        "gold": targets,
        "answer": answer,
        "final_score": final_score,
        "search_calls": row.get("search_calls") or [],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze reference rollout JSONL files.")
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--relaxed_f1_threshold", type=float, default=0.6)
    args = parser.parse_args()

    report = analyze(args.jsonl, relaxed_f1_threshold=args.relaxed_f1_threshold)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
