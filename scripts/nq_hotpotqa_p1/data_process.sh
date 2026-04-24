#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/nq_hotpotqa_p1}

mkdir -p "$LOCAL_DIR"

TRAIN_DATA=${TRAIN_DATA:-nq,hotpotqa}
TEST_DATA=${TEST_DATA:-nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle}

python "$WORK_DIR/scripts/data_process/qa_search_train_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources "$TRAIN_DATA"

python "$WORK_DIR/scripts/data_process/qa_search_test_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources "$TEST_DATA"
