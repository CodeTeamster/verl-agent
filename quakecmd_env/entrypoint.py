#!/usr/bin/env python3
"""Quake driver: run the repository's ALFWorld GRPO launcher."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _rank() -> int:
    """Return the launcher rank, accepting the names used by Quake launchers."""
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def main() -> None:
    # veRL creates and owns its Ray cluster. Exactly one launcher process must
    # therefore become its driver.
    if _rank() != 0:
        print(f"Skip ALFWorld Ray driver on launcher rank {_rank()}.", flush=True)
        return

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "quakecmd_env" / "run_alfworld.sh"
    if not (repo_root / "verl").is_dir() or not launcher.is_file():
        raise RuntimeError(f"Incomplete verl-agent checkout under {repo_root}")

    os.chdir(repo_root)
    os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"
    # Requirements are installed by Quake before the entry starts.  Install
    # this checkout only after its actual root is known; using -e from the
    # requirements file is unreliable because Quake may use another CWD.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo_root)],
        check=True,
    )
    # The base image includes DeepSpeed, but this FSDP + vLLM training path
    # does not use it. Transformers imports an installed DeepSpeed eagerly;
    # that import initializes Triton in the CPU-only Ray TaskRunner, where Ray
    # intentionally hides CUDA_VISIBLE_DEVICES, and fails before workers start.
    if importlib.util.find_spec("deepspeed") is not None:
        print("Removing unused base-image deepspeed to keep CPU Ray actors CUDA-independent.", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "deepspeed"],
            check=True,
        )
    # Quake's training_start_params become Hydra overrides after the engine.
    raise SystemExit(subprocess.call(["bash", str(launcher), "vllm", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
