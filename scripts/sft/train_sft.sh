#!/usr/bin/env bash
set -euo pipefail

DATA_NAME=${DATA_NAME:-nq_hotpotqa_p1}
DATA_DIR=${DATA_DIR:-data/${DATA_NAME}}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-3B-Instruct}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATA_NAME}-search-p1-sft-qwen2.5-3b-it-format}
PROJECT_NAME=${PROJECT_NAME:-Search-P1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=${N_GPUS:-4}

TRAIN_SOURCE=${TRAIN_SOURCE:-${DATA_DIR}/train.parquet}
VAL_SOURCE=${VAL_SOURCE:-${DATA_DIR}/test.parquet}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_p1_sft_train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_p1_sft_val.parquet}
BUILD_DATA=${BUILD_DATA:-1}
TRAIN_LIMIT=${TRAIN_LIMIT:-10000}
VAL_LIMIT=${VAL_LIMIT:-1000}
MAX_QUERY_WORDS=${MAX_QUERY_WORDS:-18}
SEED=${SEED:-7}

MAX_LENGTH=${MAX_LENGTH:-2048}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
VALIDATE_BEFORE_TRAINING=${VALIDATE_BEFORE_TRAINING:-true}

LOG_DIR=${LOG_DIR:-logs}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_NAME}}
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

if [[ "${BUILD_DATA}" == "1" ]]; then
  python scripts/sft/build_sft.py \
    --input "${TRAIN_SOURCE}" \
    --output "${TRAIN_FILE}" \
    --output-format verl_parquet \
    --limit "${TRAIN_LIMIT}" \
    --max-query-words "${MAX_QUERY_WORDS}" \
    --seed "${SEED}" \
    --conversation-format single_assistant \
    --shuffle

  python scripts/sft/build_sft.py \
    --input "${VAL_SOURCE}" \
    --output "${VAL_FILE}" \
    --output-format verl_parquet \
    --limit "${VAL_LIMIT}" \
    --max-query-words "${MAX_QUERY_WORDS}" \
    --seed "${SEED}" \
    --conversation-format single_assistant \
    --shuffle
fi

PYTHONUNBUFFERED=1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${N_GPUS}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size="${MICRO_BATCH_SIZE}" \
  data.balance_dp_token=false \
  model.partial_pretrain="${BASE_MODEL}" \
  model.enable_gradient_checkpointing=true \
  model.trust_remote_code=false \
  model.fsdp_config.cpu_offload=false \
  model.fsdp_config.offload_params=false \
  optim.lr="${LR}" \
  optim.warmup_steps_ratio=0.1 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger=['swanlab'] \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.validate_before_training="${VALIDATE_BEFORE_TRAINING}" \
  2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
