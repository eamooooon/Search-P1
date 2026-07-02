#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# HotpotQA P1 evaluation (val_only)
#   - GRPO (adv_estimator=grpo, no critic) to align with train_grpo.sh
#   - 4 GPUs, path-reward / planner-format on by default
#   - Default: run on 100 validation samples for a quick effect + speed check
# ---------------------------------------------------------------------------

data_name=hotpotqa_p1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_DIR=${DATA_DIR:-data/${data_name}}

# How many GPUs are visible -> trainer.n_gpus_per_node
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

BASE_MODEL=${BASE_MODEL:-}
if [ -z "$BASE_MODEL" ]; then
    echo "Please set BASE_MODEL to the model or checkpoint path to evaluate."
    echo "  e.g. BASE_MODEL=checkpoints/hotpotqa_p1-grpo-qwen2.5-3b-v22-b/actor/global_step_30"
    exit 1
fi

# Tunable knobs (override from shell if you want a different sample size etc.)
VAL_DATA_NUM=${VAL_DATA_NUM:-1500}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-384}
MAX_TURNS=${MAX_TURNS:-4}
TOPK=${TOPK:-3}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}

# Path-reward knobs (mirroring train_grpo.sh)
SELF_CONSISTENCY_WEIGHT=${SELF_CONSISTENCY_WEIGHT:-0.05}
REFERENCE_ALIGNMENT_WEIGHT=${REFERENCE_ALIGNMENT_WEIGHT:-0.05}
PATH_MATCH_STRATEGY=${PATH_MATCH_STRATEGY:-intent_lexical}
REQUIRE_SEARCH_FOR_FORMAT=${REQUIRE_SEARCH_FOR_FORMAT:-true}
MAX_PLAN_STEPS=${MAX_PLAN_STEPS:-4}
MAX_REFERENCE_STEPS=${MAX_REFERENCE_STEPS:-4}

WAND_PROJECT=${WAND_PROJECT:-Search-P1}
# Derive a name from the checkpoint path so multiple eval runs don't collide.
CKPT_TAG=$(echo "$BASE_MODEL" | sed -e 's|/$||' -e 's|.*checkpoints/||' -e 's|/actor/global_step_|-step|' -e 's|/|-|g')
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval-${data_name}-${CKPT_TAG}-n${VAL_DATA_NUM}}

mkdir -p logs

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

echo "============================================================"
echo "  Eval experiment : $EXPERIMENT_NAME"
echo "  Checkpoint      : $BASE_MODEL"
echo "  GPUs            : $CUDA_VISIBLE_DEVICES  (n=$N_GPUS)"
echo "  Val samples     : $VAL_DATA_NUM  (batch=$VAL_BATCH_SIZE)"
echo "  Retriever       : $RETRIEVER_URL  (topk=$TOPK, max_turns=$MAX_TURNS)"
echo "  Trajectory dump : logs/${EXPERIMENT_NAME}.jsonl"
echo "  Console log     : logs/${EXPERIMENT_NAME}.log"
echo "============================================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=128 \
    data.val_data_num=${VAL_DATA_NUM} \
    data.train_batch_size=128 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['swanlab'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.save_freq=999999 \
    trainer.test_freq=999999 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    reward_model.path_match_strategy=${PATH_MATCH_STRATEGY} \
    reward_model.require_search_for_format=${REQUIRE_SEARCH_FOR_FORMAT} \
    reward_model.max_plan_steps=${MAX_PLAN_STEPS} \
    reward_model.max_reference_steps=${MAX_REFERENCE_STEPS} \
    reward_model.self_consistency_weight=${SELF_CONSISTENCY_WEIGHT} \
    reward_model.reference_alignment_weight=${REFERENCE_ALIGNMENT_WEIGHT} \
    reward_model.trajectory_dump_path=logs/${EXPERIMENT_NAME}.jsonl \
    reward_model.trajectory_dump_limit=${VAL_DATA_NUM} \
    max_turns=${MAX_TURNS} \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=${TOPK} \
    2>&1 | tee logs/${EXPERIMENT_NAME}.log
