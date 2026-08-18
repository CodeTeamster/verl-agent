#!/usr/bin/env bash
set -euo pipefail
# Match the vLLM mode used by the training launcher.
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ -d /home/jovyan/ssd/yrc ]; then
    export STORAGE_ROOT=/home/jovyan/ssd/yrc
else
    export STORAGE_ROOT=/home/jovyan/nas/yrc
fi
export TMPDIR=${STORAGE_ROOT}/tmp/verl-agent-eval
export RAY_TMPDIR=${STORAGE_ROOT}/tmp/ray-eval
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
export HF_HOME="${STORAGE_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ALFWORLD_DATA="${STORAGE_ROOT}/dataset/alfworld"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 CHECKPOINT_PATH [seen|unseen]" >&2
    exit 2
fi

case "${2:-seen}" in
    seen) eval_dataset=eval_in_distribution ;;
    unseen) eval_dataset=eval_out_of_distribution ;;
    *) echo "Split must be seen or unseen" >&2; exit 2 ;;
esac

val_data_size=140
val_batch_size=140
enforce_eager=false
gpu_memory_utilization=0.7
max_num_batched_tokens=32768

gpus_suspended=false
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [ "$gpus_suspended" = true ]; then
        gpu-dryrun up 1,2,3 || echo "Warning: failed to restore GPU state" >&2
    fi
    rm -rf -- "$TMPDIR" || echo "Warning: failed to clean ${TMPDIR}" >&2
    # Ray may still be flushing logs after ray.shutdown().
    for _ in {1..5}; do
        rm -rf -- "$RAY_TMPDIR" 2>/dev/null || true
        [ ! -e "$RAY_TMPDIR" ] && break
        sleep 1
    done
    [ ! -e "$RAY_TMPDIR" ] || echo "Warning: failed to clean ${RAY_TMPDIR}" >&2
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

val_parquet="./assets/datasets/alfworld_eval_val${val_data_size}/text/test.parquet"
if [ -f "$val_parquet" ]; then
    echo "Reusing evaluation placeholder: ${val_parquet}"
else
    echo "Creating evaluation placeholder: ${val_parquet}"
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "./assets/datasets/alfworld_eval_val${val_data_size}" \
        --train_data_size 1 \
        --val_data_size "$val_data_size"
fi

gpu-dryrun down
gpus_suspended=true

python3 examples/evaluation/eval_alfworld.py \
    "evaluation.checkpoint_path=$(realpath "$1")" \
    "data.val_files=$(realpath "$val_parquet")" \
    "data.val_batch_size=$val_batch_size" \
    "actor_rollout_ref.model.path=$(realpath "${STORAGE_ROOT}/model/Qwen/Qwen2.5-1.5B-Instruct")" \
    "actor_rollout_ref.rollout.enforce_eager=${enforce_eager}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}" \
    "env.alfworld.eval_dataset=$eval_dataset" \
    trainer.n_gpus_per_node=4 \
    +ray_init._temp_dir="$RAY_TMPDIR"
