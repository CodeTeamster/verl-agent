set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
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
if [ -z "${TMPDIR:-}" ]; then
    if [ "$STORAGE_ROOT" = "/home/jovyan/nas/yrc" ] && [ -d /home/jovyan/ssd/yrc ]; then
        export TMPDIR=/home/jovyan/ssd/yrc/tmp
    else
        export TMPDIR=${STORAGE_ROOT}/tmp
    fi
fi
TMP_ROOT=$TMPDIR
mkdir -p "$TMP_ROOT"

LOG_ROOT=outputs/train_logs/alfworld_ppo
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${LOG_ROOT}/${RUN_ID}
mkdir -p "$RUN_DIR"
LOG_FILE=${RUN_DIR}/train.log
export TENSORBOARD_DIR=${RUN_DIR}/tensorboard
RUN_TMPDIR=$(mktemp -d "${TMP_ROOT%/}/verl-agent-alfworld-${RUN_ID}-XXXXXX") || exit 1
export TMPDIR=$RUN_TMPDIR
# Ray uses Unix sockets, whose paths must stay within the 107-byte limit.
export RAY_TMPDIR=${RAY_TMPDIR:-${TMP_ROOT}/ray}
# Fast Downward loads temporary shared libraries. Keep them on local tmpfs so
# NFS cannot leave .nfs* files that prevent RUN_TMPDIR cleanup at shutdown.
export FAST_DOWNWARD_TMPDIR=/dev/shm/verl-agent-fast-downward-${RUN_ID}
mkdir -p "$RAY_TMPDIR" "$FAST_DOWNWARD_TMPDIR"
cleanup_run_tmpdir() {
    local exit_status=$?
    trap - EXIT
    if [ -d "$FAST_DOWNWARD_TMPDIR" ]; then
        rm -rf -- "$FAST_DOWNWARD_TMPDIR" || echo "Warning: failed to clean ${FAST_DOWNWARD_TMPDIR}" >&2
    fi
    if [ -d "$RUN_TMPDIR" ]; then
        rm -rf -- "$RUN_TMPDIR" || echo "Warning: failed to clean ${RUN_TMPDIR}" >&2
    fi
    exit "$exit_status"
}

trap cleanup_run_tmpdir EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging stdout/stderr to ${LOG_FILE}"
echo "Saving tensorboard logs to ${TENSORBOARD_DIR}"

num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=128 # match GRPO and GiGPO configuration (16 × 8)
val_data_size=32
ppo_micro_batch_size_per_gpu=1 # Per-GPU micro batch size for actor, critic, rollout log-prob, and ref log-prob.
val_only=True

case "$val_only" in
    True|true)
        train_data_size=4
        val_data_size=4
        rollout_enforce_eager=True
        trainer_val_before_train=True
        trainer_val_only=True
        actor_use_kl_loss=False
        algorithm_adv_estimator=grpo
        trainer_save_freq=-1
        trainer_test_freq=-1
        ;;
    False|false)
        rollout_enforce_eager=False
        trainer_val_before_train=False
        trainer_val_only=False
        actor_use_kl_loss=True
        algorithm_adv_estimator=gae
        trainer_save_freq=10
        trainer_test_freq=10
        ;;
    *)
        echo "val_only must be True or False, got: $val_only" >&2
        exit 2
        ;;
esac

# We only use data preparation to indicate the modality and the data size.
DATASET_ROOT="./assets/datasets/alfworld_train${train_data_size}_val${val_data_size}"
DATASET_DIR="${DATASET_ROOT}/text"
TRAIN_PARQUET="${DATASET_DIR}/train.parquet"
VAL_PARQUET="${DATASET_DIR}/test.parquet"

if [ -f "$TRAIN_PARQUET" ] && [ -f "$VAL_PARQUET" ]; then
    echo "Skipping data preparation: reusing ${DATASET_DIR}."
else
    python3 -m examples.data_preprocess.prepare \
        --mode "text" \
        --local_dir "$DATASET_ROOT" \
        --train_data_size "$train_data_size" \
        --val_data_size "$val_data_size"
fi

MODEL_PATH=${STORAGE_ROOT}/model/Qwen/Qwen2.5-1.5B-Instruct
export ALFWORLD_DATA=${STORAGE_ROOT}/dataset/alfworld/

gpu-dryrun down

train_status=0
python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_PARQUET \
    data.val_files=$VAL_PARQUET \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_data_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$actor_use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$rollout_enforce_eager \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.model.path=${MODEL_PATH} \
    critic.model.use_remove_padding=True \
    critic.model.attn_implementation=flash_attention_2 \
    critic.model.fsdp_config.model_dtype=bf16 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.adv_estimator=$algorithm_adv_estimator \
    algorithm.use_kl_in_reward=False \
    ray_init.include_dashboard=False \
    +ray_init._temp_dir=${RAY_TMPDIR} \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='ppo_qwen2.5_1.5b' \
    trainer.default_local_dir=${RUN_DIR}/checkpoints \
    trainer.n_gpus_per_node=$num_devices \
    trainer.nnodes=1 \
    trainer.save_freq=$trainer_save_freq \
    trainer.test_freq=$trainer_test_freq \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.keep_best_actor_ckpt=True \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_epochs=150 \
    trainer.val_before_train=$trainer_val_before_train \
    hydra.run.dir=${RUN_DIR}/hydra \
    trainer.val_only=$trainer_val_only "$@" || train_status=$?

gpu_status=0
gpu-dryrun up 1,2,3 || gpu_status=$?


if [ "$train_status" -ne 0 ]; then
    exit "$train_status"
fi
exit "$gpu_status"
