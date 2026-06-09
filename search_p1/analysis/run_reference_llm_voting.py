import argparse
import json
import os

from search_p1.analysis.reference_io import append_jsonl, read_jsonl, write_jsonl
from search_p1.analysis.reference_llm import run_llm_voting


def _completed_custom_ids(path):
    if not path or not os.path.exists(path):
        return set()
    completed = set()
    for row in read_jsonl(path):
        custom_id = row.get("custom_id")
        if custom_id:
            completed.add(custom_id)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vote_requests")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "120")))
    parser.add_argument("--sleep", type=float, default=float(os.environ.get("LLM_SLEEP", "0")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_reference_steps", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--progress_every", type=int, default=int(os.environ.get("LLM_PROGRESS_EVERY", "25")))
    parser.add_argument("--failures_output", default=os.environ.get("LLM_FAILURES_OUTPUT"))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("LLM_RESUME", "1") != "0")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("LLM_API_KEY or --api_key is required")
    if not args.model:
        raise ValueError("LLM_MODEL or --model is required")

    if not args.resume and os.path.exists(args.output):
        os.remove(args.output)
    if args.failures_output and not args.resume and os.path.exists(args.failures_output):
        os.remove(args.failures_output)

    completed = _completed_custom_ids(args.output) if args.resume else set()
    if completed:
        print(f"resume enabled: skipping {len(completed)} completed custom_id rows", flush=True)

    def on_success(row):
        append_jsonl(args.output, row)

    def on_failure(row):
        if args.failures_output:
            append_jsonl(args.failures_output, row)

    rows, stats = run_llm_voting(
        read_jsonl(args.vote_requests),
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        sleep=args.sleep,
        limit=args.limit,
        max_reference_steps=args.max_reference_steps,
        max_retries=args.max_retries,
        progress_every=args.progress_every,
        skip_custom_ids=completed,
        on_success=on_success,
        on_failure=on_failure,
    )
    stats["output"] = args.output
    if args.failures_output:
        stats["failures_output"] = args.failures_output
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
