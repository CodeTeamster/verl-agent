# verl-agent current environment setup

This file keeps only the final commands for reproducing the current working Conda environment on this host.

## Paths

```bash
export WORKDIR=/home/jovyan/ssd/yrc/workspace/verl-agent
export CONDA_ROOT=/home/jovyan/ssd/yrc/miniconda3
export TMPDIR=/home/jovyan/ssd/yrc/tmp
export PIP_CACHE_DIR=/home/jovyan/ssd/yrc/tmp/pip-cache
export RAY_TMPDIR=/home/jovyan/ssd/yrc/ray
export FAST_DOWNWARD_TMPDIR=/home/jovyan/ssd/yrc/tmp/fast_downward_libs
export ALFWORLD_DATA=/home/jovyan/ssd/yrc/dataset/alfworld/

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$RAY_TMPDIR" "$FAST_DOWNWARD_TMPDIR"
```

## Create Conda Environment

```bash
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -n verl-agent python=3.12 pip setuptools wheel -y
conda activate verl-agent
```

## Install GPU PyTorch

```bash
python -m pip install \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Install Runtime Dependencies

```bash
python -m pip install \
  vllm==0.8.5.post1 \
  transformers==4.51.1 \
  xformers==0.0.29.post2 \
  tensordict==0.8.3 \
  tensorboard==2.16.2 \
  "ray[default]==2.43.0" \
  wandb==0.16.6 \
  google-api-core==2.19.2 \
  opentelemetry-exporter-prometheus==0.47b0 \
  flash-attn==2.7.4.post1
```

## Install Current Project

```bash
cd "$WORKDIR"
python -m pip install -e .
```

## Runtime Environment

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export TMPDIR=/home/jovyan/ssd/yrc/tmp
export PIP_CACHE_DIR=/home/jovyan/ssd/yrc/tmp/pip-cache
export RAY_TMPDIR=/home/jovyan/ssd/yrc/ray
export FAST_DOWNWARD_TMPDIR=/home/jovyan/ssd/yrc/tmp/fast_downward_libs
export ALFWORLD_DATA=/home/jovyan/ssd/yrc/dataset/alfworld/
```

## Verify Installation

```bash
python - <<'PY'
import importlib.metadata as md
import torch
import ray
import transformers
import vllm
import xformers
import tensordict

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0))
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("xformers", xformers.__version__)
print("tensordict", tensordict.__version__)
print("flash_attn", md.version("flash_attn"))
PY
```

```bash
python - <<'PY'
from flash_attn import flash_attn_func
import torch

q = torch.randn(2, 64, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
y = flash_attn_func(q, q, q, causal=True)
y.float().sum().backward()
torch.cuda.synchronize()
print("flash_attn_gpu_ok", tuple(y.shape))
PY
```

```bash
python -m py_compile \
  verl/trainer/main_ppo.py \
  verl/workers/fsdp_workers.py \
  agent_system/environments/env_package/alfworld/envs.py
```

## Run ALFWorld PPO

The current script uses:

- Qwen model: `/home/jovyan/ssd/yrc/model/Qwen/Qwen2.5-1.5B-Instruct`
- ALFWorld data: `/home/jovyan/ssd/yrc/dataset/alfworld/`
- 4 GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- FlashAttention 2
- vLLM V0
- logs under `outputs/train_logs/alfworld_ppo/<timestamp>/`
- checkpoints every 5 steps under `outputs/train_logs/alfworld_ppo/<timestamp>/checkpoints/`

```bash
cd "$WORKDIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate verl-agent

ray stop --force
bash examples/ppo_trainer/run_alfworld.sh
```

## Short Smoke Test

```bash
cd "$WORKDIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate verl-agent

ray stop --force
bash examples/ppo_trainer/run_alfworld.sh vllm \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  env.max_steps=1
```
