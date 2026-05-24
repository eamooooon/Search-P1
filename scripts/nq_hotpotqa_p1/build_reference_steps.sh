#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-$WORK_DIR/data/nq_hotpotqa_p1}

mkdir -p "$LOCAL_DIR"

if [[ -z "${TRAJECTORY_JSONL:-}" ]]; then
    echo "TRAJECTORY_JSONL is required, for example:" >&2
    echo "  TRAJECTORY_JSONL=logs/<experiment>-trajectories.jsonl bash scripts/nq_hotpotqa_p1/build_reference_steps.sh" >&2
    echo "Generate it first with train_grpo.sh or another rollout that writes reward_model.trajectory_dump_path." >&2
    exit 1
fi

if [[ ! -f "$TRAJECTORY_JSONL" ]]; then
    echo "TRAJECTORY_JSONL does not exist: $TRAJECTORY_JSONL" >&2
    echo "Run a trajectory-producing smoke run first. For P1 GRPO, train_grpo.sh writes:" >&2
    echo "  logs/<experiment>-trajectories.jsonl" >&2
    exit 1
fi

REFERENCE_STEPS_OUTPUT=${REFERENCE_STEPS_OUTPUT:-$LOCAL_DIR/reference_steps.jsonl}
VOTE_REQUESTS_OUTPUT=${VOTE_REQUESTS_OUTPUT:-$LOCAL_DIR/reference_vote_requests.jsonl}
LLM_VOTES_FILE=${LLM_VOTES_FILE:-}
if [[ -n "$LLM_VOTES_FILE" ]]; then
    WRITE_VOTE_REQUESTS=${WRITE_VOTE_REQUESTS:-0}
else
    WRITE_VOTE_REQUESTS=${WRITE_VOTE_REQUESTS:-1}
fi
MIN_SUCCESSFUL=${MIN_SUCCESSFUL:-1}
MAX_SUCCESSFUL_PER_QUESTION=${MAX_SUCCESSFUL_PER_QUESTION:-64}
MIN_VOTE_COUNT=${MIN_VOTE_COUNT:-2}
MIN_VOTE_RATIO=${MIN_VOTE_RATIO:-0.2}
MAX_REFERENCE_STEPS=${MAX_REFERENCE_STEPS:-4}

ARGS=(
    "$TRAJECTORY_JSONL"
    --output "$REFERENCE_STEPS_OUTPUT"
    --min_successful "$MIN_SUCCESSFUL"
    --max_successful_per_question "$MAX_SUCCESSFUL_PER_QUESTION"
    --min_vote_count "$MIN_VOTE_COUNT"
    --min_vote_ratio "$MIN_VOTE_RATIO"
    --max_reference_steps "$MAX_REFERENCE_STEPS"
)

if [[ "$WRITE_VOTE_REQUESTS" == "1" ]]; then
    ARGS+=(--vote_requests "$VOTE_REQUESTS_OUTPUT")
fi

if [[ -n "$LLM_VOTES_FILE" ]]; then
    ARGS+=(--llm_votes "$LLM_VOTES_FILE")
fi

cd "$WORK_DIR"
python -m search_p1.analysis.build_reference_steps "${ARGS[@]}"

echo "Wrote reference steps to $REFERENCE_STEPS_OUTPUT"
if [[ "$WRITE_VOTE_REQUESTS" == "1" ]]; then
    echo "Wrote LLM vote requests to $VOTE_REQUESTS_OUTPUT"
else
    echo "Skipped rewriting LLM vote requests because LLM_VOTES_FILE is set"
fi
