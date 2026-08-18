set -euo pipefail
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ -z "${STORAGE_ROOT:-}" ]; then
    if [ -d /home/jovyan/ssd/yrc ]; then
        export STORAGE_ROOT=/home/jovyan/ssd/yrc
    else
        export STORAGE_ROOT=/home/jovyan/nas/yrc
    fi
fi
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ALFWORLD_DATA=${STORAGE_ROOT}/dataset/alfworld/

if [ -z "${TMPDIR:-}" ]; then
    if [ "$STORAGE_ROOT" = "/home/jovyan/nas/yrc" ] && [ -d /home/jovyan/ssd/yrc ]; then
        export TMPDIR=/home/jovyan/ssd/yrc/tmp
    else
        export TMPDIR=${STORAGE_ROOT}/tmp
    fi
fi
TMP_ROOT=$TMPDIR
mkdir -p "$TMP_ROOT"

LOG_ROOT=outputs/train_logs/alfworld_gigpo
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${LOG_ROOT}/${RUN_ID}
mkdir -p "$RUN_DIR"
LOG_FILE=${RUN_DIR}/train.log
export TENSORBOARD_DIR=${RUN_DIR}/tensorboard

export TMPDIR=${TMP_ROOT%/}/g
export RAY_TMPDIR=${RAY_TMPDIR:-${TMP_ROOT}/ray}
export FAST_DOWNWARD_TMPDIR=/dev/shm/verl-agent-fast-downward-${RUN_ID}
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$FAST_DOWNWARD_TMPDIR"

cleanup_run_tmpdir() {
    local exit_status=$?
    trap - EXIT
    if [ -d "$FAST_DOWNWARD_TMPDIR" ]; then
        rm -rf -- "$FAST_DOWNWARD_TMPDIR" || echo "Warning: failed to clean ${FAST_DOWNWARD_TMPDIR}" >&2
    fi
}
trap cleanup_run_tmpdir EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging stdout/stderr to ${LOG_FILE}"
echo "Saving tensorboard logs to ${TENSORBOARD_DIR}"

num_devices=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
train_data_size=16
val_data_size=128
group_size=8
envs_per_worker=4
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
MODEL_PATH=${STORAGE_ROOT}/model/Qwen/Qwen2.5-1.5B-Instruct

DATASET_ROOT="./assets/datasets/alfworld_gigpo_train${train_data_size}_val${val_data_size}"
DATASET_DIR="${DATASET_ROOT}/text"
TRAIN_PARQUET="${DATASET_DIR}/train.parquet"
VAL_PARQUET="${DATASET_DIR}/test.parquet"
if [ -f "$TRAIN_PARQUET" ] && [ -f "$VAL_PARQUET" ]; then
    echo "Skipping data preparation: reusing ${DATASET_DIR}."
else
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$DATASET_ROOT" \
        --train_data_size "$train_data_size" \
        --val_data_size "$val_data_size"
fi

gpu-dryrun down

train_status=0
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_PARQUET \
    data.val_files=$VAL_PARQUET \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    +ray_init._temp_dir=${RAY_TMPDIR} \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.alfworld.envs_per_worker=${envs_per_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.default_local_dir=${RUN_DIR}/checkpoints \
    trainer.n_gpus_per_node=$num_devices \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    hydra.run.dir=${RUN_DIR}/hydra $@

gpu_status=0
gpu-dryrun up 1,2,3 || gpu_status=$?

if [ "$train_status" -ne 0 ]; then
    exit "$train_status"
fi
exit "$gpu_status"