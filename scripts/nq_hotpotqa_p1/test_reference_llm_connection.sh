#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

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

: "${LLM_API_KEY:?LLM_API_KEY is required. Put it in .env.llm/.env or set it in the environment.}"
: "${LLM_MODEL:?LLM_MODEL is required. Put it in .env.llm/.env or set it in the environment.}"

cd "$WORK_DIR"
python -m search_p1.analysis.test_reference_llm_connection
