#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/hotpotqa_p1}

mkdir -p "$LOCAL_DIR"

TRAIN_DATA=${TRAIN_DATA:-hotpotqa}
TEST_DATA=${TEST_DATA:-2wikimultihopqa,musique,bamboogle}
MAX_REFERENCE_STEPS=${MAX_REFERENCE_STEPS:-4}
REFERENCE_ONLY=${REFERENCE_ONLY:-0}
REFERENCE_ONLY_SPLITS=${REFERENCE_ONLY_SPLITS:-train}
DEFAULT_REFERENCE_STEPS_FILE=$LOCAL_DIR/reference_steps_v22_corrected_llm.jsonl
if [[ -z "${REFERENCE_STEPS_FILE:-}" && -f "$DEFAULT_REFERENCE_STEPS_FILE" ]]; then
    REFERENCE_STEPS_FILE=$DEFAULT_REFERENCE_STEPS_FILE
fi

REFERENCE_ARGS=()
TRAIN_REFERENCE_ARGS=()
TEST_REFERENCE_ARGS=()
if [[ -n "${REFERENCE_STEPS_FILE:-}" ]]; then
    echo "DATA_PROCESS_REFERENCE_STEPS file=$REFERENCE_STEPS_FILE max_reference_steps=$MAX_REFERENCE_STEPS"
    REFERENCE_ARGS+=(--reference_steps_file "$REFERENCE_STEPS_FILE")
    REFERENCE_ARGS+=(--max_reference_steps "$MAX_REFERENCE_STEPS")
fi
if [[ "$REFERENCE_ONLY" == "1" || "$REFERENCE_ONLY" == "true" || "$REFERENCE_ONLY" == "True" ]]; then
    echo "DATA_PROCESS_REFERENCE_ONLY enabled splits=$REFERENCE_ONLY_SPLITS"
    if [[ ",$REFERENCE_ONLY_SPLITS," == *",all,"* || ",$REFERENCE_ONLY_SPLITS," == *",train,"* ]]; then
        TRAIN_REFERENCE_ARGS+=(--reference_only)
    fi
    if [[ ",$REFERENCE_ONLY_SPLITS," == *",all,"* || ",$REFERENCE_ONLY_SPLITS," == *",test,"* ]]; then
        TEST_REFERENCE_ARGS+=(--reference_only)
    fi
fi

python "$WORK_DIR/scripts/data_process/qa_search_train_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources "$TRAIN_DATA" \
    "${REFERENCE_ARGS[@]}" \
    "${TRAIN_REFERENCE_ARGS[@]}"

python "$WORK_DIR/scripts/data_process/qa_search_test_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources "$TEST_DATA" \
    "${REFERENCE_ARGS[@]}" \
    "${TEST_REFERENCE_ARGS[@]}"
