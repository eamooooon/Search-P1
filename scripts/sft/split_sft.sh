#!/usr/bin/env bash
set -euo pipefail

DATA_NAME=${DATA_NAME:-nq_hotpotqa_p1}
DATA_DIR=${DATA_DIR:-data/${DATA_NAME}}
INPUT=${INPUT:-${DATA_DIR}/search_p1_sft_format_10k.parquet}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${DATA_DIR}/search_p1_sft_train.parquet}
VAL_OUTPUT=${VAL_OUTPUT:-${DATA_DIR}/search_p1_sft_val.parquet}
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-7}
FORMAT=${FORMAT:-auto}

args=(
  --input "${INPUT}"
  --train-output "${TRAIN_OUTPUT}"
  --val-output "${VAL_OUTPUT}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
  --format "${FORMAT}"
)

if [[ -n "${VAL_SIZE:-}" ]]; then
  args+=(--val-size "${VAL_SIZE}")
fi

if [[ "${SHUFFLE:-1}" == "0" ]]; then
  args+=(--no-shuffle)
fi

python scripts/sft/split_sft.py "${args[@]}"
