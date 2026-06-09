#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/nq_hotpotqa_p1}
LIMIT=${LIMIT:-}

ARGS=()
if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

cd "$WORK_DIR"
python -m search_p1.analysis.check_reference_steps "$LOCAL_DIR/train.parquet" "${ARGS[@]}"
python -m search_p1.analysis.check_reference_steps "$LOCAL_DIR/test.parquet" "${ARGS[@]}"
