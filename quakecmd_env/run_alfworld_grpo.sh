#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then shift; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${TRAINER_NNODES:?Set TRAINER_NNODES in quakecmd --envs}"
: "${GPU_PER_POD:?Set GPU_PER_POD in quakecmd --envs}"
: "${MODEL_PATH:?Set MODEL_PATH in quakecmd --envs to your model path}"
: "${RUN_DIR:?Set RUN_DIR in quakecmd --envs}"
: "${PLACEHOLDER_DATA_PATH:?Set PLACEHOLDER_DATA_PATH in quakecmd --envs}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA in quakecmd --envs to the uploaded ALFWorld assets}"
: "${GEOMETRY3K_DATA:?Set GEOMETRY3K_DATA in quakecmd --envs to the local Geometry3K parquet cache}"

if [ ! -d "$MODEL_PATH" ] || [ ! -d "$ALFWORLD_DATA" ] || [ ! -d "$GEOMETRY3K_DATA" ]; then
    echo "ERROR: MODEL_PATH, ALFWORLD_DATA, and GEOMETRY3K_DATA must be mounted directories." >&2
    exit 2
fi

# Preserve the official runner's training defaults.  These are only made
# configurable to fit the platform's assigned topology.
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-2}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
SAVE_FREQ=${SAVE_FREQ:--1}
ENVS_PER_WORKER=${ENVS_PER_WORKER:-1}

if (( GPU_PER_POD % VLLM_TP_SIZE != 0 )); then
    echo "ERROR: GPU_PER_POD=${GPU_PER_POD} must be divisible by VLLM_TP_SIZE=${VLLM_TP_SIZE}." >&2
    exit 2
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export ALFWORLD_DATA
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PLACEHOLDER="${PLACEHOLDER_DATA_PATH}/alfworld_grpo_train${TRAIN_DATA_SIZE}_val${VAL_DATA_SIZE}"
TRAIN_PARQUET="${PLACEHOLDER}/text/train.parquet"
VAL_PARQUET="${PLACEHOLDER}/text/test.parquet"
mkdir -p "$RUN_DIR" "$PLACEHOLDER"

echo "================== Train preflight =================="
awk '
  /MemTotal/ {total = $2}
  /MemAvailable/ {available = $2}
  END {
    print "MEMORY_TOTAL_GIB=" sprintf("%.2f", total / 1024 / 1024)
    print "MEMORY_AVAILABLE_GIB=" sprintf("%.2f", available / 1024 / 1024)
  }
' /proc/meminfo
echo "ALGORITHM=grpo"
echo "ALFWORLD_ACCELERATOR=${ALFWORLD_ACCELERATOR:-nvidia}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "TRAINER_NNODES=${TRAINER_NNODES} GPU_PER_POD=${GPU_PER_POD} VLLM_TP_SIZE=${VLLM_TP_SIZE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ALFWORLD_DATA=${ALFWORLD_DATA}"
echo "GEOMETRY3K_DATA=${GEOMETRY3K_DATA}"
echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE} VAL_DATA_SIZE=${VAL_DATA_SIZE} GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} ENVS_PER_WORKER=${ENVS_PER_WORKER} ENFORCE_EAGER=${ENFORCE_EAGER} SAVE_FREQ=${SAVE_FREQ}"
for name in LAUNCH_RAY RAY_ADDRESS VLLM_USE_V1; do
    if [ -n "${!name:-}" ]; then
        echo "${name}=${!name}"
    fi
done
python3 - <<'PY'
import ray

ray.init(address="auto", logging_level="ERROR")
for node in ray.nodes():
    if node["Alive"]:
        resources = node["Resources"]
        print(f"RAY_NODE={node['NodeManagerAddress']} CPU={resources.get('CPU', 0):g} GPU={resources.get('GPU', 0):g}")
ray.shutdown()
PY
nvidia-smi

if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$VAL_PARQUET" ]; then
    python3 "${REPO_ROOT}/examples/data_preprocess/prepare.py" \
        --mode text \
        --local_dir "${PLACEHOLDER}" \
        --dataset_dir "${GEOMETRY3K_DATA}" \
        --train_data_size "${TRAIN_DATA_SIZE}" \
        --val_data_size "${VAL_DATA_SIZE}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=${TRAIN_DATA_SIZE} \
    data.val_batch_size=${VAL_DATA_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${VLLM_TP_SIZE} \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=${GRPO_GROUP_SIZE} \
    env.alfworld.envs_per_worker=${ENVS_PER_WORKER} \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=grpo_qwen2.5_1.5b \
    trainer.default_local_dir="${RUN_DIR}/ckpts" \
    trainer.validation_metrics_csv="${RUN_DIR}/validation_metrics.csv" \
    trainer.n_gpus_per_node=${GPU_PER_POD} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.keep_best_actor_ckpt=True \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    hydra.run.dir="${RUN_DIR}/hydra" \
    "$@"
