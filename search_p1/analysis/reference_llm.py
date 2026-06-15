import json
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start:end + 1])


def chat_completion(base_url, api_key, model, messages, temperature, timeout, response_format="json"):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def extract_paper_reasoning_path(text):
    match = re.search(
        r"<correct_reasoning_path>(.*?)</correct_reasoning_path>",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("No correct_reasoning_path block found in LLM response")
    block = match.group(1)
    steps = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[\.)]|[-*])\s*", "", line).strip()
        if line:
            steps.append(line)
    return {"reference_steps": steps}


def valid_reference_steps(value, max_reference_steps):
    if not isinstance(value, list):
        return []
    steps = []
    seen = set()
    for step in value:
        if not isinstance(step, str):
            continue
        step = " ".join(step.strip().split())
        if not step or "<" in step or ">" in step or "http://" in step or "https://" in step:
            continue
        normalized_placeholder = step.lower().strip()
        normalized_placeholder = re.sub(r"^(?:search\s+)?(?:query|search query|concrete search query|step)\s*\d*\.?$", "", normalized_placeholder)
        if not normalized_placeholder:
            continue
        if not step.lower().startswith("search "):
            step = "Search " + step
        search_body = step[len("Search "):].strip()
        if len(re.sub(r"[^A-Za-z0-9]", "", search_body)) < 3:
            continue
        key = step.lower()
        if key in seen:
            continue
        seen.add(key)
        steps.append(step)
        if len(steps) >= max_reference_steps:
            break
    return steps


def candidate_action_fallback_steps(parsed, max_reference_steps):
    candidate_actions = parsed.get("candidate_actions") if isinstance(parsed, dict) else None
    if not isinstance(candidate_actions, list):
        return []
    actions = []
    for item in candidate_actions:
        if isinstance(item, dict):
            action = item.get("action")
        else:
            action = item
        if isinstance(action, str):
            actions.append(action)
    return valid_reference_steps(actions, max_reference_steps)


def parse_reference_response(content, response_format):
    if response_format == "paper":
        return extract_paper_reasoning_path(content)
    return extract_json_object(content)


def process_vote_request(request,
                         base_url,
                         api_key,
                         model,
                         temperature,
                         timeout,
                         max_reference_steps,
                         max_retries,
                         response_format):
    custom_id = request.get("custom_id")
    content = None
    parsed = None
    request_start = time.time()
    for attempt in range(max_retries):
        try:
            content = chat_completion(
                base_url,
                api_key,
                model,
                request["messages"],
                temperature,
                timeout,
                response_format=response_format,
            )
            parsed = parse_reference_response(content, response_format)
            steps = valid_reference_steps(parsed.get("reference_steps"), max_reference_steps)
            reference_step_source = "llm"
            if not steps:
                steps = candidate_action_fallback_steps(parsed, max_reference_steps)
                reference_step_source = "candidate_actions_fallback"
            if not steps:
                raise ValueError("LLM response has no valid reference_steps")
            output = dict(request.get("metadata") or {})
            output.update({
                "custom_id": custom_id,
                "reference_steps": steps,
                "reference_step_source": reference_step_source,
                "vote_model": model,
            })
            return {
                "ok": True,
                "row": output,
                "latency": time.time() - request_start,
                "attempts": attempt + 1,
                "custom_id": custom_id,
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, KeyError, json.JSONDecodeError) as exc:
            if attempt + 1 >= max_retries:
                failure = {
                    "custom_id": custom_id,
                    "error": str(exc),
                    "attempts": max_retries,
                }
                if content is not None:
                    failure["raw_content"] = content
                if parsed is not None:
                    failure["parsed_response"] = parsed
                return {
                    "ok": False,
                    "row": failure,
                    "latency": time.time() - request_start,
                    "attempts": max_retries,
                    "custom_id": custom_id,
                }
            time.sleep(min(2 ** attempt, 10))


def run_llm_voting(requests,
                   base_url,
                   api_key,
                   model,
                   temperature=0,
                   timeout=120,
                   sleep=0,
                   limit=None,
                   max_reference_steps=4,
                   max_retries=3,
                   progress_every=25,
                   skip_custom_ids=None,
                   on_success=None,
                   on_failure=None,
                   response_format="json",
                   workers=1):
    if workers < 1:
        raise ValueError("workers must be >= 1")
    rows = []
    failures = 0
    processed = 0
    skipped = 0
    latencies = []
    start_time = time.time()
    skip_custom_ids = skip_custom_ids or set()
    pending_requests = []
    for index, request in enumerate(requests):
        if limit is not None and index >= limit:
            break
        custom_id = request.get("custom_id")
        if custom_id in skip_custom_ids:
            skipped += 1
            continue
        pending_requests.append(request)

    def handle_result(result):
        nonlocal failures
        latencies.append(result["latency"])
        if result["ok"]:
            rows.append(result["row"])
            if on_success is not None:
                on_success(result["row"])
        else:
            failures += 1
            if on_failure is not None:
                on_failure(result["row"])
            print(f"failed custom_id={result['custom_id']}: {result['row']['error']}", flush=True)

    def maybe_print_progress(force=False):
        if not progress_every:
            return
        if not force and processed % progress_every != 0:
            return
        elapsed = max(time.time() - start_time, 1e-6)
        rate = processed / elapsed
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        print(
            f"progress processed={processed} success={len(rows)} failures={failures} "
            f"skipped={skipped} workers={workers} rate={rate:.2f}/s "
            f"avg_latency={avg_latency:.2f}s elapsed={elapsed:.1f}s",
            flush=True,
        )

    if workers == 1:
        for request in pending_requests:
            result = process_vote_request(
                request,
                base_url,
                api_key,
                model,
                temperature,
                timeout,
                max_reference_steps,
                max_retries,
                response_format,
            )
            processed += 1
            handle_result(result)
            maybe_print_progress()
            if sleep:
                time.sleep(sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_request = {
                executor.submit(
                    process_vote_request,
                    request,
                    base_url,
                    api_key,
                    model,
                    temperature,
                    timeout,
                    max_reference_steps,
                    max_retries,
                    response_format,
                ): request
                for request in pending_requests
            }
            for future in as_completed(future_to_request):
                processed += 1
                result = future.result()
                handle_result(result)
                maybe_print_progress()
                if sleep:
                    time.sleep(sleep)

    elapsed = max(time.time() - start_time, 1e-6)
    maybe_print_progress(force=True)
    return rows, {
        "vote_requests": processed,
        "vote_rows": len(rows),
        "failures": failures,
        "skipped": skipped,
        "workers": workers,
        "elapsed": elapsed,
        "rate": processed / elapsed,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }
