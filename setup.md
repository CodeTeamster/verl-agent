# verl-agent current environment setup

This file keeps only the final commands for reproducing the current working Conda environment on this host.

## Paths

```bash
if [ -d /home/jovyan/ssd/yrc ]; then
  STORAGE_ROOT=/home/jovyan/ssd/yrc
else
  STORAGE_ROOT=/home/jovyan/nas/yrc
fi
# If both roots exist, set STORAGE_ROOT manually before continuing.

WORKDIR="$STORAGE_ROOT/workspace/verl-agent"
CONDA_ROOT="$STORAGE_ROOT/miniconda3"
export TMPDIR="$STORAGE_ROOT/tmp"
export PIP_CACHE_DIR="$STORAGE_ROOT/tmp/pip-cache"
export XDG_CACHE_HOME="$STORAGE_ROOT/cache"
export HF_HOME="$STORAGE_ROOT/cache/huggingface"
export RAY_TMPDIR="$STORAGE_ROOT/ray"
export FAST_DOWNWARD_TMPDIR="$STORAGE_ROOT/tmp/fast_downward_libs"
ALFWORLD_DATA="$STORAGE_ROOT/dataset/alfworld/"
MODEL_DIR="$STORAGE_ROOT/model/Qwen/Qwen2.5-1.5B-Instruct"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$RAY_TMPDIR" "$FAST_DOWNWARD_TMPDIR" "$ALFWORLD_DATA" "$MODEL_DIR"
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
  opentelemetry-exporter-prometheus==0.47b0
```

## Install FlashAttention

Install `flash-attn` after PyTorch. If `nvcc` is not already available on the host, install the CUDA compiler into the Conda environment first.

```bash
python -m pip install ninja packaging

if ! command -v nvcc >/dev/null 2>&1; then
  conda install -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4 -y
fi

MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

## Install Current Project

```bash
cd "$WORKDIR"
python -m pip install -e .
```

## Install ALFWorld CLI

`alfworld-download` is a console script installed by the `alfworld` Python package. Install ALFWorld before downloading the task data.

```bash
python -m pip install \
  gymnasium==0.29.1 \
  stable-baselines3==2.6.0 \
  alfworld
```

## Download Assets

Download ALFWorld PDDL files, game files, and the pre-trained MaskRCNN detector. `alfworld-download` is run with `XDG_CACHE_HOME` on the selected storage root; sync the downloaded files into `ALFWORLD_DATA` for the project.

```bash
mkdir -p "$ALFWORLD_DATA"
alfworld-download -f
rsync -a "$XDG_CACHE_HOME/alfworld/" "$ALFWORLD_DATA"/
```

Download the Qwen model checkpoint.

```bash
hf-download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir "$MODEL_DIR"
```

## Key Component Checks

```bash
python - <<'PY'
import importlib.metadata as md
import torch
import ray
import transformers
import vllm
import xformers
import tensordict
import gymnasium
import alfworld

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0))
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("xformers", xformers.__version__)
print("tensordict", tensordict.__version__)
print("gymnasium", gymnasium.__version__)
print("alfworld", md.version("alfworld"))
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
