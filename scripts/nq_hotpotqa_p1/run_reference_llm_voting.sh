#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/nq_hotpotqa_p1}

if [[ -z "${LLM_ENV_FILE:-}" ]]; then
    if [[ -f "$WORK_DIR/.env.llm" ]]; then
        LLM_ENV_FILE="$WORK_DIR/.env.llm"
    elif [[ -f "$WORK_DIR/.env" ]]; then
        LLM_ENV_FILE="$WORK_DIR/.env"
    fi
fi

if [[ -n "${LLM_ENV_FILE:-}" ]]; then
    echo "Loading LLM config from $LLM_ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$LLM_ENV_FILE"
    set +a
fi

VOTE_REQUESTS_FILE=${VOTE_REQUESTS_FILE:-$LOCAL_DIR/reference_vote_requests.jsonl}
LLM_VOTES_OUTPUT=${LLM_VOTES_OUTPUT:-$LOCAL_DIR/reference_vote_results.jsonl}
LLM_FAILURES_OUTPUT=${LLM_FAILURES_OUTPUT:-$LOCAL_DIR/reference_vote_failures.jsonl}
LLM_LOG=${LLM_LOG:-$LOCAL_DIR/reference_llm_voting.log}
MAX_REFERENCE_STEPS=${MAX_REFERENCE_STEPS:-4}
LLM_LIMIT=${LLM_LIMIT:-200}
LLM_PROGRESS_EVERY=${LLM_PROGRESS_EVERY:-25}

: "${LLM_API_KEY:?LLM_API_KEY is required. Put it in .env.llm/.env or set it in the environment.}"
: "${LLM_MODEL:?LLM_MODEL is required. Put it in .env.llm/.env or set it in the environment.}"
export LLM_FAILURES_OUTPUT
export LLM_PROGRESS_EVERY

ARGS=(
    "$VOTE_REQUESTS_FILE"
    --output "$LLM_VOTES_OUTPUT"
    --max_reference_steps "$MAX_REFERENCE_STEPS"
)

if [[ -n "$LLM_LIMIT" ]]; then
    ARGS+=(--limit "$LLM_LIMIT")
fi

cd "$WORK_DIR"
mkdir -p "$(dirname "$LLM_LOG")"
python -m search_p1.analysis.run_reference_llm_voting "${ARGS[@]}" 2>&1 | tee "$LLM_LOG"

echo "Wrote LLM voting results to $LLM_VOTES_OUTPUT"
echo "Wrote LLM voting failures to $LLM_FAILURES_OUTPUT"
echo "Wrote LLM voting log to $LLM_LOG"
