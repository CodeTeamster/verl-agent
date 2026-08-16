#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then shift; fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

: "${TRAINER_NNODES:?Set TRAINER_NNODES in quakecmd --envs}"
: "${GPU_PER_POD:?Set GPU_PER_POD in quakecmd --envs}"
: "${STORAGE_ROOT:?Set STORAGE_ROOT in quakecmd --envs to your Jindo path}"
: "${MODEL_PATH:?Set MODEL_PATH in quakecmd --envs to your model path}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist or is not mounted: $MODEL_PATH" >&2
    exit 2
fi
: "${RUN_DIR:?Set RUN_DIR in quakecmd --envs}"
: "${PLACEHOLDER_DATA_PATH:?Set PLACEHOLDER_DATA_PATH in quakecmd --envs}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA in quakecmd --envs to the uploaded ALFWorld assets}"
if [ ! -d "$ALFWORLD_DATA" ]; then
    echo "ERROR: ALFWORLD_DATA does not exist or is not mounted: $ALFWORLD_DATA" >&2
    exit 2
fi
: "${GEOMETRY3K_DATA:?Set GEOMETRY3K_DATA in quakecmd --envs}"
if [ ! -d "$GEOMETRY3K_DATA" ]; then
    echo "ERROR: GEOMETRY3K_DATA does not exist or is not mounted: $GEOMETRY3K_DATA" >&2
    exit 2
fi

export ALFWORLD_DATA="${ALFWORLD_DATA}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-32}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
ENV_CPU=${ENV_CPU:-0.05}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}

RUN_DIR="${RUN_DIR}"
PLACEHOLDER="${PLACEHOLDER_DATA_PATH}/alfworld_grpo_train${TRAIN_DATA_SIZE}_val${VAL_DATA_SIZE}"
TRAIN_PARQUET="${PLACEHOLDER}/text/train.parquet"
VAL_PARQUET="${PLACEHOLDER}/text/test.parquet"
MODEL_PATH="${MODEL_PATH}"
mkdir -p "$RUN_DIR" "$PLACEHOLDER"

echo "======= CUDA preflight ======="
python3 - <<'PY'
import ray

ray.init(address="auto", logging_level="ERROR")
for node in ray.nodes():
    if node["Alive"]:
        resources = node["Resources"]
        print(
            f"RAY_NODE={node['NodeManagerAddress']} "
            f"CPU={resources.get('CPU', 0):g} "
            f"GPU={resources.get('GPU', 0):g}"
        )
ray.shutdown()
PY
awk '/MemTotal/ {printf "MEMORY_TOTAL_GIB=%.2f\n", $2 / 1024 / 1024}' /proc/meminfo
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable; the NVIDIA driver is not mounted in this container." >&2
    exit 2
fi
nvidia-smi
python3 - <<'PY'
import os
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("ERROR: PyTorch cannot access a CUDA device in this container.")
print(f"transformers={transformers.__version__}")
PY

# 1. Prepare placeholder parquet
if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$VAL_PARQUET" ]; then
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "${PLACEHOLDER}" \
        --dataset_dir "${GEOMETRY3K_DATA}" \
        --train_data_size "${TRAIN_DATA_SIZE}" \
        --val_data_size "${VAL_DATA_SIZE}"
fi

# 2. Train
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.standard_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_DATA_SIZE * GRPO_GROUP_SIZE)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    ray_init.include_dashboard=False \
    +ray_init.address=auto \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=${GRPO_GROUP_SIZE} \
    env.resources_per_worker.num_cpus=${ENV_CPU} \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=grpo_qwen2.5_1.5b \
    trainer.default_local_dir="${RUN_DIR}/ckpts" \
    trainer.n_gpus_per_node=${GPU_PER_POD} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    hydra.run.dir="${RUN_DIR}/hydra" \
    "$@"
