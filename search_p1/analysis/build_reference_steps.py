import argparse
import json

from search_p1.analysis.reference_io import write_jsonl
from search_p1.analysis.reference_sampling import collect_successful_groups
from search_p1.analysis.reference_voting import (
    build_reference_rows,
    build_vote_request,
    load_llm_votes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vote_requests", default=None)
    parser.add_argument("--llm_votes", default=None)
    parser.add_argument("--min_successful", type=int, default=1)
    parser.add_argument("--max_successful_per_question", type=int, default=64)
    parser.add_argument("--min_vote_count", type=int, default=2)
    parser.add_argument("--min_vote_ratio", type=float, default=0.2)
    parser.add_argument("--max_reference_steps", type=int, default=4)
    args = parser.parse_args()

    groups, stats = collect_successful_groups(
        args.trajectory_jsonl,
        max_successful_per_question=args.max_successful_per_question,
    )

    if args.vote_requests:
        requests = [
            build_vote_request(key, group)
            for key, group in groups.items()
            if group["correct"] >= args.min_successful and group["trajectories"]
        ]
        write_jsonl(args.vote_requests, requests)
    else:
        requests = []

    llm_votes = load_llm_votes(args.llm_votes)
    rows, reference_stats = build_reference_rows(
        groups,
        llm_votes=llm_votes,
        min_successful=args.min_successful,
        min_vote_count=args.min_vote_count,
        min_vote_ratio=args.min_vote_ratio,
        max_reference_steps=args.max_reference_steps,
        return_stats=True,
    )
    write_jsonl(args.output, rows)

    stats.update({
        "reference_rows": len(rows),
        "vote_request_rows": len(requests),
        "llm_vote_rows": len(llm_votes),
        **reference_stats,
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
