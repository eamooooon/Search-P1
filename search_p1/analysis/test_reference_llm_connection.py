import argparse
import json
import os
import sys
import urllib.error

from search_p1.analysis.reference_llm import (
    chat_completion,
    extract_json_object,
    valid_reference_steps,
)


def _sample_messages():
    user_prompt = {
        "question": "Who discovered radium?",
        "candidate_actions": [
            {"action": "Marie Curie radium discovery", "count": 3},
            {"action": "who discovered radium", "count": 2},
            {"action": "radium element discoverer", "count": 1},
        ],
        "instruction": (
            "Select the minimal consensus search-intent steps needed to solve the question. "
            "Return JSON only with key reference_steps. Each step must be a short 'Search ...' intent. "
            "Do not include fallback branches, exhaustive lists, URLs, or trajectory tags."
        ),
    }
    return [
        {
            "role": "system",
            "content": "You generate concise reference search plans for QA reasoning trajectories.",
        },
        {
            "role": "user",
            "content": json.dumps(user_prompt, ensure_ascii=False),
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "120")))
    parser.add_argument("--max_reference_steps", type=int, default=int(os.environ.get("MAX_REFERENCE_STEPS", "4")))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("LLM_API_KEY or --api_key is required")
    if not args.model:
        raise ValueError("LLM_MODEL or --model is required")

    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print("request=sample reference voting prompt")

    try:
        content = chat_completion(
            args.base_url,
            args.api_key,
            args.model,
            _sample_messages(),
            args.temperature,
            args.timeout,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"status=HTTPError {exc.code}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"status=ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("raw_content:")
    print(content)

    try:
        parsed = extract_json_object(content)
    except Exception as exc:
        print(f"status=CONNECTED_BUT_JSON_PARSE_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    steps = valid_reference_steps(
        parsed.get("reference_steps"),
        max_reference_steps=args.max_reference_steps,
    )
    print("parsed_json:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("validated_reference_steps:")
    print(json.dumps(steps, ensure_ascii=False, indent=2))

    if not steps:
        print("status=CONNECTED_BUT_NO_VALID_REFERENCE_STEPS", file=sys.stderr)
        return 3

    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
