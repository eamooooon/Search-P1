#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$WORK_DIR"
RUN_START_TS=$(date +%s)
SLICED_TRAIN_FILE=""

print_collect_elapsed() {
    local status=$?
    local end_ts
    local elapsed
    end_ts=$(date +%s)
    elapsed=$((end_ts - RUN_START_TS))
    if [[ -n "${SLICED_TRAIN_FILE:-}" && -f "$SLICED_TRAIN_FILE" ]]; then
        rm -f "$SLICED_TRAIN_FILE"
    fi
    echo "COLLECT_REFERENCE_DONE status=$status elapsed=${elapsed}s output=${TRAJECTORY_DUMP_PATH:-unset}"
}
trap print_collect_elapsed EXIT

data_name=hotpotqa_p1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_DIR=${DATA_DIR:-data/${data_name}}
export BASE_MODEL=${BASE_MODEL:-checkpoints/hotpotqa_p1-grpo-qwen2.5-3b-v20/actor/global_step_50}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${data_name}-reference-vllm-v22-3}

TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-38400}
TRAIN_DATA_OFFSET=${TRAIN_DATA_OFFSET:-76800}
TRAIN_DATA_SLICE=${TRAIN_DATA_SLICE:-1}
RESAMPLE_INDEX_FILE=${RESAMPLE_INDEX_FILE:-}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-384}
TOTAL_STEPS_WAS_SET=${TOTAL_STEPS+x}
TOTAL_STEPS=${TOTAL_STEPS:-100}
TRAINER_TOTAL_STEPS=$((TOTAL_STEPS + 1))
N_AGENT=${N_AGENT:-16}
MAX_TURNS=${MAX_TURNS:-4}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-800}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}

TRAJECTORY_DUMP_PATH=${TRAJECTORY_DUMP_PATH:-logs/$EXPERIMENT_NAME.jsonl}
TRAJECTORY_DUMP_LIMIT=${TRAJECTORY_DUMP_LIMIT:--1}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

mkdir -p logs "$(dirname "$TRAJECTORY_DUMP_PATH")"

TRAIN_FILE=$DATA_DIR/train.parquet
EFFECTIVE_TRAIN_DATA_NUM=$TRAIN_DATA_NUM

if [[ -n "$RESAMPLE_INDEX_FILE" ]]; then
    if [[ ! -f "$RESAMPLE_INDEX_FILE" ]]; then
        echo "RESAMPLE_INDEX_FILE does not exist: $RESAMPLE_INDEX_FILE" >&2
        exit 1
    fi
    SLICED_TRAIN_FILE=$(mktemp "$DATA_DIR/reference_resample_XXXXXX.parquet")
    RESAMPLE_COUNT_FILE=$(mktemp "$DATA_DIR/reference_resample_count_XXXXXX.txt")
    python3 - "$TRAIN_FILE" "$SLICED_TRAIN_FILE" "$RESAMPLE_INDEX_FILE" "$RESAMPLE_COUNT_FILE" <<'PY'
import json
import sys

import pandas as pd

input_path, output_path, index_path, count_path = sys.argv[1:5]


def read_requested_ids(path):
    requested = set()
    requested_meta = set()
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                index = row.get("index")
                if index is None:
                    raise SystemExit(f"missing index in {path}:{line_number}")
                requested.add(str(index))
                data_source = row.get("data_source")
                split = row.get("split")
                if data_source is not None and split is not None:
                    requested_meta.add((str(data_source), str(split), str(index)))
            else:
                requested.add(line)
    return requested, requested_meta


requested_ids, requested_meta = read_requested_ids(index_path)
if not requested_ids:
    raise SystemExit(f"no ids found in {index_path}")

frame = pd.read_parquet(input_path)


def row_index(row):
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        return str(extra_info.get("index"))
    row_id = row.get("id")
    if isinstance(row_id, str) and row_id.startswith("train_"):
        return row_id[len("train_"):]
    return None


def row_meta_key(row):
    extra_info = row.get("extra_info")
    split = extra_info.get("split") if isinstance(extra_info, dict) else None
    index = row_index(row)
    data_source = row.get("data_source")
    if data_source is None or split is None or index is None:
        return None
    return (str(data_source), str(split), str(index))


if requested_meta:
    mask = frame.apply(lambda row: row_meta_key(row) in requested_meta, axis=1)
else:
    mask = frame.apply(lambda row: row_index(row) in requested_ids, axis=1)

subset = frame[mask].reset_index(drop=True)
if subset.empty:
    raise SystemExit(
        f"empty resample subset: ids={len(requested_ids)}, dataset_len={len(frame)}"
    )

found_ids = set()
for _, row in subset.iterrows():
    index = row_index(row)
    if index is not None:
        found_ids.add(index)
missing = sorted(requested_ids - found_ids, key=lambda item: int(item) if item.isdigit() else item)
if missing:
    preview = ", ".join(missing[:10])
    raise SystemExit(
        f"missing {len(missing)} requested ids in train parquet; first missing ids: {preview}"
    )

subset.to_parquet(output_path, index=False)
with open(count_path, "w", encoding="utf-8") as f:
    f.write(str(len(subset)))
print(
    f"COLLECT_REFERENCE_RESAMPLE input={input_path} output={output_path} "
    f"index_file={index_path} requested={len(requested_ids)} actual={len(subset)} "
    f"dataset_len={len(frame)}",
    flush=True,
)
PY
    RESAMPLE_ACTUAL_COUNT=$(cat "$RESAMPLE_COUNT_FILE")
    rm -f "$RESAMPLE_COUNT_FILE"
    TRAIN_FILE=$SLICED_TRAIN_FILE
    EFFECTIVE_TRAIN_DATA_NUM=null
    TRAIN_DATA_SLICE=0
    if [[ -z "$TOTAL_STEPS_WAS_SET" ]]; then
        TOTAL_STEPS=$(( (RESAMPLE_ACTUAL_COUNT + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
        TRAINER_TOTAL_STEPS=$((TOTAL_STEPS + 1))
    fi
elif [[ "$TRAIN_DATA_SLICE" == "1" ]]; then
    if [[ -z "$TRAIN_DATA_NUM" || "$TRAIN_DATA_NUM" == "null" ]]; then
        echo "TRAIN_DATA_NUM must be a positive integer when TRAIN_DATA_SLICE=1" >&2
        exit 1
    fi
    SLICED_TRAIN_FILE=$(mktemp "$DATA_DIR/reference_slice_offset_${TRAIN_DATA_OFFSET}_XXXXXX.parquet")
    python3 - "$TRAIN_FILE" "$SLICED_TRAIN_FILE" "$TRAIN_DATA_OFFSET" "$TRAIN_DATA_NUM" <<'PY'
import sys
import pandas as pd

input_path, output_path, offset_text, count_text = sys.argv[1:5]
offset = int(offset_text)
count = int(count_text)
frame = pd.read_parquet(input_path)
if offset < 0:
    raise SystemExit("TRAIN_DATA_OFFSET must be >= 0")
if count <= 0:
    raise SystemExit("TRAIN_DATA_NUM must be > 0")
subset = frame.iloc[offset:offset + count].reset_index(drop=True)
if subset.empty:
    raise SystemExit(f"empty slice: offset={offset}, count={count}, dataset_len={len(frame)}")
subset.to_parquet(output_path, index=False)
print(
    f"COLLECT_REFERENCE_SLICE input={input_path} output={output_path} "
    f"offset={offset} requested={count} actual={len(subset)} dataset_len={len(frame)}",
    flush=True,
)
PY
    TRAIN_FILE=$SLICED_TRAIN_FILE
    EFFECTIVE_TRAIN_DATA_NUM=null
fi

echo "COLLECT_REFERENCE_CONFIG train_file=$TRAIN_FILE train_data_num=$EFFECTIVE_TRAIN_DATA_NUM train_data_offset=$TRAIN_DATA_OFFSET train_data_slice=$TRAIN_DATA_SLICE resample_index_file=${RESAMPLE_INDEX_FILE:-none} train_batch_size=$TRAIN_BATCH_SIZE n_agent=$N_AGENT total_steps=$TOTAL_STEPS max_turns=$MAX_TURNS output=$TRAJECTORY_DUMP_PATH"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$TRAIN_FILE \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=$EFFECTIVE_TRAIN_DATA_NUM \
    data.val_data_num=1 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=$MAX_OBS_LENGTH \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=$N_AGENT \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['swanlab'] \
    +trainer.rollout_only=true \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.project_name=Search-P1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TRAINER_TOTAL_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    reward_model.path_match_strategy=intent_lexical \
    reward_model.require_search_for_format=true \
    reward_model.max_plan_steps=4 \
    reward_model.max_reference_steps=4 \
    reward_model.self_consistency_weight=0 \
    reward_model.reference_alignment_weight=0 \
    reward_model.trajectory_dump_path=$TRAJECTORY_DUMP_PATH \
    reward_model.trajectory_dump_limit=$TRAJECTORY_DUMP_LIMIT \
    reward_model.trajectory_dump_full_solution=false \
    max_turns=$MAX_TURNS \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

echo "Wrote trajectory dump to $TRAJECTORY_DUMP_PATH"
