import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from search_p1.analysis.analyze_reference_rollouts import (
    final_answer,
    gold_in_information,
    relaxed_answer_match,
    targets_from_row,
)
from search_p1.analysis.reference_io import key_metadata, read_jsonl, row_key
from search_p1.analysis.reward_format import qa_em_format


def is_correct(row, mode, relaxed_f1_threshold):
    reward_components = row.get("reward_components") or {}
    answer = final_answer(row)
    targets = targets_from_row(row)
    strict = bool(answer and targets and qa_em_format.em_check(answer, targets))

    if mode == "strict":
        return strict
    if mode == "score1":
        return reward_components.get("final_score") == 1.0
    if mode == "relaxed":
        return strict or relaxed_answer_match(answer, targets, relaxed_f1_threshold)
    if mode == "usable":
        return (
            strict
            or relaxed_answer_match(answer, targets, relaxed_f1_threshold)
            or gold_in_information(row, targets)
        )
    raise ValueError(f"unknown correct mode: {mode}")


def export_resample_questions(paths, output, min_correct, mode, relaxed_f1_threshold):
    counts = defaultdict(Counter)
    metadata = {}
    global_stats = Counter()

    for path in paths:
        for row in read_jsonl(path):
            global_stats["rows"] += 1
            key = row_key(row)
            if key is None:
                global_stats["skipped_no_key"] += 1
                continue
            if key not in metadata:
                metadata[key] = key_metadata(row)
            counts[key]["total"] += 1
            if is_correct(row, mode, relaxed_f1_threshold):
                counts[key]["correct"] += 1

    rows = []
    for key, count in counts.items():
        if count["correct"] >= min_correct:
            continue
        meta = dict(metadata.get(key) or {})
        meta.update({
            "correct_count": count["correct"],
            "total_count": count["total"],
            "min_correct": min_correct,
            "correct_mode": mode,
            "key": list(key),
        })
        rows.append(meta)

    rows.sort(key=lambda row: (
        int(row["correct_count"]),
        -int(row["total_count"]),
        str(row.get("data_source")),
        str(row.get("split")),
        int(row["index"]) if str(row.get("index", "")).isdigit() else str(row.get("index")),
    ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_questions = len(counts)
    summary = {
        "inputs": [str(path) for path in paths],
        "rows": global_stats["rows"],
        "questions": total_questions,
        "skipped_no_key": global_stats["skipped_no_key"],
        "correct_mode": mode,
        "min_correct": min_correct,
        "enough_questions": total_questions - len(rows),
        "resample_questions": len(rows),
        "thresholds": {
            ">=1": sum(1 for count in counts.values() if count["correct"] >= 1),
            ">=2": sum(1 for count in counts.values() if count["correct"] >= 2),
            ">=3": sum(1 for count in counts.values() if count["correct"] >= 3),
            ">=5": sum(1 for count in counts.values() if count["correct"] >= 5),
        },
        "output": str(output),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Export question ids whose collected correct rollouts are below a threshold."
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_correct", type=int, default=3)
    parser.add_argument(
        "--correct_mode",
        choices=("score1", "strict", "relaxed", "usable"),
        default="score1",
        help="score1 uses reward_components.final_score == 1.0; usable also accepts relaxed answer/evidence-gold rows.",
    )
    parser.add_argument("--relaxed_f1_threshold", type=float, default=0.6)
    args = parser.parse_args()

    summary = export_resample_questions(
        paths=args.jsonl,
        output=args.output,
        min_correct=args.min_correct,
        mode=args.correct_mode,
        relaxed_f1_threshold=args.relaxed_f1_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
