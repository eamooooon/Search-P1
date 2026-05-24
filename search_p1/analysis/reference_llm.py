import json
import time
import urllib.error
import urllib.request


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


def chat_completion(base_url, api_key, model, messages, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
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
        if not step.lower().startswith("search "):
            step = "Search " + step
        key = step.lower()
        if key in seen:
            continue
        seen.add(key)
        steps.append(step)
        if len(steps) >= max_reference_steps:
            break
    return steps


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
                   on_failure=None):
    rows = []
    failures = 0
    processed = 0
    skipped = 0
    start_time = time.time()
    skip_custom_ids = skip_custom_ids or set()
    for index, request in enumerate(requests):
        if limit is not None and index >= limit:
            break
        custom_id = request.get("custom_id")
        if custom_id in skip_custom_ids:
            skipped += 1
            continue
        processed += 1

        for attempt in range(max_retries):
            try:
                content = chat_completion(
                    base_url,
                    api_key,
                    model,
                    request["messages"],
                    temperature,
                    timeout,
                )
                parsed = extract_json_object(content)
                steps = valid_reference_steps(parsed.get("reference_steps"), max_reference_steps)
                if not steps:
                    raise ValueError("LLM response has no valid reference_steps")
                output = dict(request.get("metadata") or {})
                output.update({
                    "custom_id": custom_id,
                    "reference_steps": steps,
                    "vote_model": model,
                })
                rows.append(output)
                if on_success is not None:
                    on_success(output)
                break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt + 1 >= max_retries:
                    failures += 1
                    failure = {
                        "custom_id": custom_id,
                        "error": str(exc),
                        "attempts": max_retries,
                    }
                    if on_failure is not None:
                        on_failure(failure)
                    print(f"failed custom_id={custom_id}: {exc}", flush=True)
                else:
                    time.sleep(min(2 ** attempt, 10))

        if progress_every and processed % progress_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = processed / elapsed
            print(
                f"progress processed={processed} success={len(rows)} failures={failures} skipped={skipped} rate={rate:.2f}/s",
                flush=True,
            )

        if sleep:
            time.sleep(sleep)
    return rows, {
        "vote_requests": processed,
        "vote_rows": len(rows),
        "failures": failures,
        "skipped": skipped,
    }
