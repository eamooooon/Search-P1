#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/hotpotqa_p1}
PRESERVE_ENV_NAMES=(
    LLM_API_KEY
    LLM_MODEL
    LLM_BASE_URL
    LLM_TEMPERATURE
    LLM_TIMEOUT
    LLM_LIMIT
    LLM_START_OFFSET
    LLM_PROGRESS_EVERY
    LLM_WORKERS
    VOTE_PROMPT_STYLE
    LLM_RESPONSE_FORMAT
    LLM_RESUME
    LLM_FAILURES_OUTPUT
)

for name in "${PRESERVE_ENV_NAMES[@]}"; do
    if [[ -v $name ]]; then
        declare "__PRESERVE_HAS_${name}=1"
        declare "__PRESERVE_VALUE_${name}=${!name}"
    fi
done

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

for name in "${PRESERVE_ENV_NAMES[@]}"; do
    has_name="__PRESERVE_HAS_${name}"
    value_name="__PRESERVE_VALUE_${name}"
    if [[ -v $has_name ]]; then
        export "$name=${!value_name}"
    fi
done

CORRECTED_ROLLOUTS=${CORRECTED_ROLLOUTS:-$LOCAL_DIR/reference_v22_corrected_rollouts_min2.jsonl}
REFERENCE_STEPS_OUTPUT=${REFERENCE_STEPS_OUTPUT:-$LOCAL_DIR/reference_steps_v22_corrected_llm.jsonl}
LLM_LOG=${LLM_LOG:-$LOCAL_DIR/reference_v22_corrected_llm_voting.log}
LLM_FAILURES_OUTPUT=${LLM_FAILURES_OUTPUT:-$LOCAL_DIR/reference_steps_v22_corrected_llm.failures.jsonl}

MIN_SUCCESSFUL=${MIN_SUCCESSFUL:-3}
MAX_SUCCESSFUL_PER_QUESTION=${MAX_SUCCESSFUL_PER_QUESTION:-999999}
MAX_REFERENCE_STEPS=${MAX_REFERENCE_STEPS:-4}
VOTE_PROMPT_STYLE=${VOTE_PROMPT_STYLE:-consensus_json}
LLM_RESPONSE_FORMAT=${LLM_RESPONSE_FORMAT:-json}
LLM_LIMIT=${LLM_LIMIT:-}
LLM_START_OFFSET=${LLM_START_OFFSET:-0}
LLM_PROGRESS_EVERY=${LLM_PROGRESS_EVERY:-10}
LLM_WORKERS=${LLM_WORKERS:-1}

: "${LLM_API_KEY:?LLM_API_KEY is required. Put it in .env.llm/.env or set it in the environment.}"
: "${LLM_MODEL:?LLM_MODEL is required. Put it in .env.llm/.env or set it in the environment.}"

if [[ ! -f "$CORRECTED_ROLLOUTS" ]]; then
    echo "CORRECTED_ROLLOUTS does not exist: $CORRECTED_ROLLOUTS" >&2
    exit 1
fi

cd "$WORK_DIR"
mkdir -p "$LOCAL_DIR" "$(dirname "$REFERENCE_STEPS_OUTPUT")" "$(dirname "$LLM_LOG")"

{
    echo "REFERENCE_LLM_DIRECT_CONFIG input=$CORRECTED_ROLLOUTS output=$REFERENCE_STEPS_OUTPUT start_offset=$LLM_START_OFFSET limit=$LLM_LIMIT workers=$LLM_WORKERS prompt_style=$VOTE_PROMPT_STYLE response_format=$LLM_RESPONSE_FORMAT"
    ARGS=(
        "$CORRECTED_ROLLOUTS"
        --output "$REFERENCE_STEPS_OUTPUT"
        --min_successful "$MIN_SUCCESSFUL"
        --max_successful_per_question "$MAX_SUCCESSFUL_PER_QUESTION"
        --max_reference_steps "$MAX_REFERENCE_STEPS"
        --prompt_style "$VOTE_PROMPT_STYLE"
        --response_format "$LLM_RESPONSE_FORMAT"
        --start_offset "$LLM_START_OFFSET"
        --progress_every "$LLM_PROGRESS_EVERY"
        --workers "$LLM_WORKERS"
        --failures_output "$LLM_FAILURES_OUTPUT"
    )
    if [[ -n "$LLM_LIMIT" ]]; then
        ARGS+=(--limit "$LLM_LIMIT")
    fi
    python -m search_p1.analysis.build_reference_steps_llm_direct "${ARGS[@]}"
    echo "Wrote LLM reference steps to $REFERENCE_STEPS_OUTPUT"
    echo "Wrote LLM voting log to $LLM_LOG"
} 2>&1 | tee -a "$LLM_LOG"
