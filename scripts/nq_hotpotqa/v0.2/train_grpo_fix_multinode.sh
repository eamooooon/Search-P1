data_name=nq_hotpotqa_train
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
N_NODES=${N_NODES:-2}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-"http://127.0.0.1:8265"} # your head node dashboard address

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/${data_name}} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

WAND_PROJECT="Search-R1"

export BASE_MODEL=${BASE_MODEL:-${PROJECT_DIR}/models/Qwen2.5-3B-Instruct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${data_name}-search-r1-grpo-qwen2.5-3b-it-em-v0.2-fix}
RESUME_PATH=${RESUME_PATH:-${PROJECT_DIR}/checkpoints/${data_name}-search-r1-grpo-qwen2.5-3b-it-em-v0.2-fix}
SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-}
SWANLAB_RESUME=${SWANLAB_RESUME:-must}
SWANLAB_MODE=${SWANLAB_MODE:-cloud}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-eamooooon}
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-7b-it-em-multinode-${N_NODES}

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export RAY_RUNTIME_ENV_IGNORE_GITIGNORE=${RAY_RUNTIME_ENV_IGNORE_GITIGNORE:-1}

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

ulimit -n 65535
mkdir -p logs

ray job submit --address=$RAY_DASHBOARD_ADDRESS \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    SWANLAB_RUN_ID=$SWANLAB_RUN_ID \
    SWANLAB_RESUME=$SWANLAB_RESUME \
    SWANLAB_MODE=$SWANLAB_MODE \
    SWANLAB_API_KEY=$SWANLAB_API_KEY \
    SWANLAB_WORKSPACE=$SWANLAB_WORKSPACE \
    python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['swanlab'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=305 \
    trainer.resume_path=$RESUME_PATH \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJECT_DIR/checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
