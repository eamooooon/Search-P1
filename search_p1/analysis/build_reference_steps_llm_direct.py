import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from search_p1.analysis.reference_io import append_jsonl, read_jsonl, row_key
from search_p1.analysis.reference_llm import process_vote_request
from search_p1.analysis.reference_sampling import collect_successful_groups
from search_p1.analysis.reference_voting import build_vote_request


def completed_custom_ids(path):
    if not path or not os.path.exists(path):
        return set()
    completed = set()
    for row in read_jsonl(path):
        custom_id = row.get("custom_id")
        if custom_id:
            completed.add(custom_id)
            continue
        key = row_key(row)
        if key is not None:
            completed.add("|".join(key))
    return completed


def output_row(request, result, group, model):
    metadata = request.get("metadata") or {}
    row = dict(metadata)
    row.update({
        "custom_id": request.get("custom_id"),
        "reference_steps": result["row"]["reference_steps"],
        "reference_plan_source": "llm_vote",
        "reference_step_source": result["row"].get("reference_step_source", "llm"),
        "vote_model": model,
        "source_trajectory_count": group["total"],
        "accepted_trajectory_count": group["correct"],
    })
    return {key: value for key, value in row.items() if value is not None}


def process_item(item, args):
    request_index, key, group = item
    request = build_vote_request(key, group, prompt_style=args.prompt_style)
    try:
        result = process_vote_request(
            request,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_reference_steps=args.max_reference_steps,
            max_retries=args.max_retries,
            response_format=args.response_format,
        )
    except Exception as exc:
        result = {
            "ok": False,
            "row": {
                "custom_id": request.get("custom_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": 1,
            },
            "latency": 0.0,
            "attempts": 1,
            "custom_id": request.get("custom_id"),
        }
    return request_index, key, group, request, result


def main():
    parser = argparse.ArgumentParser(
        description="Build final reference_steps directly with an LLM, writing only clean successful rows.",
    )
    parser.add_argument("trajectory_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "120")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_offset", type=int, default=int(os.environ.get("LLM_START_OFFSET", "0")))
    parser.add_argument("--max_reference_steps", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--progress_every", type=int, default=int(os.environ.get("LLM_PROGRESS_EVERY", "25")))
    parser.add_argument("--min_successful", type=int, default=2)
    parser.add_argument("--max_successful_per_question", type=int, default=None)
    parser.add_argument("--prompt_style", choices=["paper", "consensus_json"], default="paper")
    parser.add_argument("--response_format", choices=["paper", "json"], default="paper")
    parser.add_argument("--failures_output", default=os.environ.get("LLM_FAILURES_OUTPUT"))
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LLM_WORKERS", "1")))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("LLM_API_KEY or --api_key is required")
    if not args.model:
        raise ValueError("LLM_MODEL or --model is required")
    if args.start_offset < 0:
        raise ValueError("--start_offset must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    start_time = time.time()
    print(
        f"LLM_DIRECT_STAGE=collect_groups input={args.trajectory_jsonl} "
        f"min_successful={args.min_successful}",
        flush=True,
    )
    groups, group_stats = collect_successful_groups(
        args.trajectory_jsonl,
        max_successful_per_question=args.max_successful_per_question,
    )
    print(json.dumps({"group_stats": group_stats}, ensure_ascii=False), flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_output = args.failures_output
    if failures_output is None:
        failures_output = str(output_path.with_suffix(output_path.suffix + ".failures"))
    Path(failures_output).parent.mkdir(parents=True, exist_ok=True)
    completed = completed_custom_ids(str(output_path))
    if completed:
        print(f"resume enabled: skipping {len(completed)} existing output rows", flush=True)

    eligible = [
        (key, group)
        for key, group in groups.items()
        if group["correct"] >= args.min_successful and group["trajectories"]
    ]
    print(
        f"LLM_DIRECT_STAGE=llm_vote eligible={len(eligible)} "
        f"start_offset={args.start_offset} limit={args.limit} workers={args.workers} output={args.output}",
        flush=True,
    )

    processed = 0
    skipped_existing = 0
    skipped_offset = 0
    written = 0
    failures = 0
    latencies = []

    selected = []
    for request_index, (key, group) in enumerate(eligible):
        if request_index < args.start_offset:
            skipped_offset += 1
            continue
        custom_id = "|".join(key)
        if custom_id in completed:
            skipped_existing += 1
            continue
        selected.append((request_index, key, group))
        if args.limit is not None and len(selected) >= args.limit:
            break

    def handle_result(request_index, group, request, result):
        nonlocal processed, written, failures
        processed += 1
        latencies.append(result["latency"])
        custom_id = request["custom_id"]

        if not result["ok"]:
            failures += 1
            failure_row = dict(request.get("metadata") or {})
            failure_row.update(result["row"])
            failure_row.update({
                "request_index": request_index,
                "custom_id": custom_id,
                "bad_row_written": False,
            })
            append_jsonl(failures_output, failure_row)
            print(
                "LLM_DIRECT_ERROR "
                f"request_index={request_index} custom_id={custom_id} "
                f"error={result['row'].get('error')} failures={failures} "
                f"failures_output={failures_output}",
                flush=True,
            )
            if args.stop_on_failure:
                print(
                    json.dumps({
                        "status": "failed",
                        "bad_row_written": False,
                        "processed": processed,
                        "written": written,
                        "failures": failures,
                        "skipped_existing": skipped_existing,
                        "skipped_offset": skipped_offset,
                        "output": args.output,
                        "failures_output": failures_output,
                    }, ensure_ascii=False, indent=2),
                    flush=True,
                )
                return 1
            return 0

        append_jsonl(str(output_path), output_row(request, result, group, args.model))
        completed.add(custom_id)
        written += 1

        if args.progress_every and processed % args.progress_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            print(
                f"progress processed={processed} written={written} skipped_existing={skipped_existing} "
                f"failures={failures} skipped_offset={skipped_offset} rate={processed / elapsed:.2f}/s "
                f"avg_latency={avg_latency:.2f}s elapsed={elapsed:.1f}s",
                flush=True,
            )
        return 0

    if args.workers == 1:
        for item in selected:
            request_index, key, group, request, result = process_item(item, args)
            exit_code = handle_result(request_index, group, request, result)
            if exit_code:
                return exit_code
    else:
        pending = set()
        selected_iter = iter(selected)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while True:
                while len(pending) < args.workers * 2:
                    try:
                        item = next(selected_iter)
                    except StopIteration:
                        break
                    pending.add(executor.submit(process_item, item, args))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    request_index, key, group, request, result = future.result()
                    exit_code = handle_result(request_index, group, request, result)
                    if exit_code:
                        for pending_future in pending:
                            pending_future.cancel()
                        return exit_code

    elapsed = max(time.time() - start_time, 1e-6)
    summary = {
        "status": "complete",
        "eligible": len(eligible),
        "selected": len(selected),
        "processed": processed,
        "written": written,
        "failures": failures,
        "skipped_existing": skipped_existing,
        "skipped_offset": skipped_offset,
        "output": args.output,
        "failures_output": failures_output,
        "workers": args.workers,
        "elapsed": elapsed,
        "rate": processed / elapsed,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
