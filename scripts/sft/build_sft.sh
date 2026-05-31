#!/usr/bin/env bash
set -euo pipefail

DATA_NAME=${DATA_NAME:-nq_hotpotqa_p1}
INPUT=${INPUT:-data/${DATA_NAME}/train.parquet}
LIMIT=${LIMIT:-10000}
MAX_QUERY_WORDS=${MAX_QUERY_WORDS:-18}
SEED=${SEED:-7}
CONVERSATION_FORMAT=${CONVERSATION_FORMAT:-multi_turn}
OUTPUT_FORMAT=verl_parquet

if [[ -z "${OUTPUT:-}" ]]; then
  if [[ "${OUTPUT_FORMAT}" == "verl_parquet" ]]; then
    OUTPUT=data/${DATA_NAME}/search_p1_sft_format_10k.parquet
  else
    OUTPUT=data/${DATA_NAME}/search_p1_sft_format_10k.jsonl
  fi
fi

python scripts/sft/build_sft.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --limit "${LIMIT}" \
  --max-query-words "${MAX_QUERY_WORDS}" \
  --seed "${SEED}" \
  --conversation-format "${CONVERSATION_FORMAT}" \
  --output-format "${OUTPUT_FORMAT}" \
  --shuffle
