import argparse
import json
import re
import sys
from collections import Counter, deque
from importlib import util
from pathlib import Path
from statistics import mean
from typing import Optional


FAILURE_REASONS = (
    "complete",
    "no_actions",
    "invalid_planner",
    "partial_plan_coverage",
    "unmatched_actions",
    "redundant_actions",
)

TRACK_A_KEYS = (
    "self_consistency",
    "self_r_planner",
    "self_n_plan",
    "self_n_actions",
    "self_n_exec",
)

ACTION_QUALITY_REASONS = (
    "plain_query",
    "bare_search",
    "search_prefix",
    "low_info_search_prefix",
    "function_search",
    "tool_call_prefix",
    "tool_response_text",
    "nested_tag",
    "json_like",
    "url",
    "overlong",
    "empty",
)


def load_reward_module(repo_root: Path):
    module_path = repo_root / "verl" / "utils" / "reward_score" / "qa_em_format.py"
    spec = util.spec_from_file_location("qa_em_format_direct", module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path, limit: Optional[int] = None, tail: Optional[int] = None):
    if limit is not None and tail is not None:
        raise ValueError("--limit and --tail are mutually exclusive")

    if tail is not None:
        rows = deque(maxlen=tail)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if stripped:
                    rows.append((line_number, stripped))
        for line_number, stripped in rows:
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
        return

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            count += 1
            if limit is not None and count >= limit:
                break


def get_solution_str(row: dict):
    for key in ("solution_str", "trajectory", "response", "text", "completion"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_assistant_content(text: str):
    marker = "<|im_start|>assistant"
    if marker in text:
        return text.rsplit(marker, 1)[1].split("<|im_end|>", 1)[0]
    return text


def extract_model_tool_calls(text: str):
    content = extract_assistant_content(text)
    content = re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)
    return [match.strip() for match in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)]


def classify_tool_call_content(query: str):
    query = query.strip() if query else ""
    if not query:
        return "empty"
    if re.search(r"</?[^>]+>", query):
        return "nested_tag"
    if re.search(r"https?://|www\.", query, re.IGNORECASE):
        return "url"
    if len(query.split()) > 32:
        return "overlong"
    if re.fullmatch(r"(?:query|search)", query, re.IGNORECASE):
        return "bare_search"
    if re.fullmatch(r"(?:search|query)-[^\s]+", query, re.IGNORECASE):
        return "low_info_search_prefix"
    if re.search(r"\btool_call\s*:?\s*search\b|^\s*tool_call\b", query, re.IGNORECASE):
        return "tool_call_prefix"
    if re.search(r"\bsearch\s*\(", query, re.IGNORECASE):
        return "function_search"
    if re.search(r"\btool_response\s*:", query, re.IGNORECASE):
        return "tool_response_text"
    if re.search(r"^\s*(?:query|search)\s*:?\s+(?!engine\b)", query, re.IGNORECASE):
        return "search_prefix"
    if (query.startswith("{") and query.endswith("}")) or (
        query.startswith("[") and query.endswith("]")
    ):
        return "json_like"
    return "plain_query"


def normalize_ground_truth(row: dict):
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        target = ground_truth["target"]
    else:
        target = row.get("target", row.get("answers", row.get("answer")))

    if target is None:
        return None
    if isinstance(target, str):
        target = [target]
    if isinstance(target, list):
        return {"target": target}
    return None


def percentile(values: list[float], q: float):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else 0.0


def classify_failure(components: dict):
    n_plan = components["self_n_plan"]
    n_actions = components["self_n_actions"]
    n_exec = components["self_n_exec"]
    if components["self_r_planner"] == 0:
        return "invalid_planner"
    if n_actions == 0:
        return "no_actions"
    if n_exec == 0:
        return "unmatched_actions"
    if n_exec < n_plan:
        return "partial_plan_coverage"
    if n_actions > n_exec:
        return "redundant_actions"
    return "complete"


def snippet(text: str, max_chars: int):
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def analyze_rows(rows, reward_module, match_strategy: str, max_plan_steps: Optional[int], low_score_threshold: float, sample_size: int):
    records = []
    failure_counts = Counter()
    action_quality_counts = Counter()
    low_samples = []
    missing_solution = 0

    for source, row in rows:
        solution_str = get_solution_str(row)
        if solution_str is None:
            missing_solution += 1
            continue
        for tool_call in extract_model_tool_calls(solution_str):
            action_quality_counts[classify_tool_call_content(tool_call)] += 1

        components = reward_module.compute_self_consistency_components(
            solution_str,
            match_strategy=match_strategy,
            max_plan_steps=max_plan_steps,
        )
        ground_truth = normalize_ground_truth(row)
        if ground_truth is not None:
            score_components = reward_module.compute_score_components(
                solution_str,
                ground_truth,
                path_match_strategy=match_strategy,
                max_plan_steps=max_plan_steps,
            )
            components["base_score"] = score_components["base_score"]
            components["final_score"] = score_components["final_score"]

        reason = classify_failure(components)
        failure_counts[reason] += 1
        record = {
            "source": source,
            "reason": reason,
            "solution_str": solution_str,
            **components,
        }
        records.append(record)

        if components["self_consistency"] <= low_score_threshold and len(low_samples) < sample_size:
            low_samples.append(record)

    return records, failure_counts, action_quality_counts, low_samples, missing_solution


def summarize(records: list[dict], failure_counts: Counter):
    summary = {"samples": len(records)}
    for key in TRACK_A_KEYS:
        values = [float(record[key]) for record in records]
        summary[key] = {
            "mean": mean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "max": max(values) if values else 0.0,
        }

    plan_coverage = [
        ratio(float(record["self_n_exec"]), float(record["self_n_plan"]))
        for record in records
    ]
    action_efficiency = [
        ratio(float(record["self_n_exec"]), float(record["self_n_actions"]))
        for record in records
    ]
    summary["planner_valid_rate"] = summary["self_r_planner"]["mean"]
    summary["mean_plan_coverage"] = mean(plan_coverage) if plan_coverage else 0.0
    summary["mean_action_efficiency"] = mean(action_efficiency) if action_efficiency else 0.0
    summary["failure_counts"] = {reason: failure_counts.get(reason, 0) for reason in FAILURE_REASONS}
    return summary


def summarize_action_quality(action_quality_counts: Counter):
    total = sum(action_quality_counts.values())
    return {
        "total_tool_calls": total,
        "counts": {
            reason: action_quality_counts.get(reason, 0)
            for reason in ACTION_QUALITY_REASONS
        },
    }


def bucket_failure_counts(records: list[dict]):
    counts = Counter(record["reason"] for record in records)
    return {reason: counts.get(reason, 0) for reason in FAILURE_REASONS}


def build_buckets(records: list[dict], bucket_size: Optional[int]):
    if bucket_size is None:
        return []

    buckets = []
    for start in range(0, len(records), bucket_size):
        bucket_records = records[start : start + bucket_size]
        if not bucket_records:
            continue
        failure_counts = Counter(record["reason"] for record in bucket_records)
        summary = summarize(bucket_records, failure_counts)
        buckets.append(
            {
                "index": len(buckets),
                "source_range": {
                    "start": bucket_records[0]["source"],
                    "end": bucket_records[-1]["source"],
                },
                "samples": summary["samples"],
                "planner_valid_rate": summary["planner_valid_rate"],
                "self_consistency_mean": summary["self_consistency"]["mean"],
                "failure_counts": bucket_failure_counts(bucket_records),
            }
        )
    return buckets


def print_buckets(buckets: list[dict]):
    print("")
    print("Buckets / Trend:")
    if not buckets:
        print("  disabled")
        return

    for bucket in buckets:
        counts = bucket["failure_counts"]
        count_text = " ".join(f"{reason}={counts[reason]}" for reason in FAILURE_REASONS)
        source_range = bucket["source_range"]
        print(
            f"  bucket={bucket['index']} source={source_range['start']}..{source_range['end']} "
            f"samples={bucket['samples']} "
            f"planner_valid_rate={bucket['planner_valid_rate']:.4f} "
            f"self_consistency_mean={bucket['self_consistency_mean']:.4f} "
            f"{count_text}"
        )


def print_summary(
    summary: dict,
    action_quality: dict,
    low_samples: list[dict],
    missing_solution: int,
    max_chars: int,
    buckets: list[dict],
):
    print(f"Samples: {summary['samples']}")
    if missing_solution:
        print(f"Skipped rows without solution text: {missing_solution}")
    print(f"Planner valid rate: {summary['planner_valid_rate']:.4f}")
    print(f"Mean plan coverage: {summary['mean_plan_coverage']:.4f}")
    print(f"Mean action efficiency: {summary['mean_action_efficiency']:.4f}")
    print("")
    print("Track A distributions:")
    for key in TRACK_A_KEYS:
        stats = summary[key]
        print(
            f"  {key}: mean={stats['mean']:.4f} min={stats['min']:.4f} "
            f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}"
        )
    print("")
    print("Failure attribution:")
    for reason, count in sorted(summary["failure_counts"].items()):
        print(f"  {reason}: {count}")

    print("")
    print("Action quality:")
    print(f"  total_tool_calls: {action_quality['total_tool_calls']}")
    for reason, count in action_quality["counts"].items():
        if count:
            print(f"  {reason}: {count}")

    print_buckets(buckets)

    if low_samples:
        print("")
        print("Low-score samples:")
        for sample in low_samples:
            print(
                f"  - {sample['source']} reason={sample['reason']} "
                f"S_self={sample['self_consistency']:.4f} "
                f"n_plan={sample['self_n_plan']} n_actions={sample['self_n_actions']} "
                f"n_exec={sample['self_n_exec']}"
            )
            print(f"    {snippet(sample['solution_str'], max_chars)}")


def positive_int(value: str):
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze Track A Self-Consistency metrics from JSONL trajectories.",
    )
    parser.add_argument("jsonl", nargs="+", type=Path, help="Input JSONL file(s).")
    row_group = parser.add_mutually_exclusive_group()
    row_group.add_argument("--limit", type=positive_int, default=None, help="Maximum non-empty rows per input file.")
    row_group.add_argument("--tail", type=positive_int, default=None, help="Read only the last N non-empty JSONL rows per input file.")
    parser.add_argument("--bucket-size", type=positive_int, default=None, help="Group analyzed records into ordered buckets of N rows.")
    parser.add_argument("--match-strategy", default="lexical", help="Path match strategy: lexical or intent_lexical.")
    parser.add_argument("--max-plan-steps", type=positive_int, default=None, help="Mark planners longer than N steps invalid when recomputing Track A.")
    parser.add_argument("--low-score-threshold", type=float, default=0.5, help="Threshold for printing low-score samples.")
    parser.add_argument("--sample-size", type=int, default=5, help="Maximum low-score samples to print.")
    parser.add_argument("--snippet-chars", type=int, default=240, help="Maximum characters per printed sample snippet.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON instead of text.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    reward_module = load_reward_module(repo_root)
    reward_module.validate_path_match_strategy(args.match_strategy)

    rows = []
    for path in args.jsonl:
        for line_number, row in iter_jsonl(path, limit=args.limit, tail=args.tail):
            rows.append((f"{path}:{line_number}", row))

    records, failure_counts, action_quality_counts, low_samples, missing_solution = analyze_rows(
        rows,
        reward_module=reward_module,
        match_strategy=args.match_strategy,
        max_plan_steps=args.max_plan_steps,
        low_score_threshold=args.low_score_threshold,
        sample_size=args.sample_size,
    )
    summary = summarize(records, failure_counts)
    action_quality = summarize_action_quality(action_quality_counts)
    buckets = build_buckets(records, args.bucket_size)

    if args.json:
        print(json.dumps(
            {
                "summary": summary,
                "action_quality": action_quality,
                "buckets": buckets,
                "missing_solution": missing_solution,
            },
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print_summary(
            summary,
            action_quality,
            low_samples,
            missing_solution,
            max_chars=args.snippet_chars,
            buckets=buckets,
        )

    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
