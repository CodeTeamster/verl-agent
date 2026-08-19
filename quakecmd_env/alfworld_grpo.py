#!/usr/bin/env python3
"""Quake driver for the unmodified official ALFWorld launchers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path


def _rank() -> int:
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def _wait_for_platform_ray(expected_gpus: int) -> None:
    import ray

    timeout_s = int(os.environ.get("RAY_CLUSTER_WAIT_S", "300"))
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        initialized = False
        try:
            ray.init(address="auto")
            initialized = True
            if ray.cluster_resources().get("GPU", 0) >= expected_gpus:
                return
        except Exception as exc:
            last_error = repr(exc)
        finally:
            if initialized:
                ray.shutdown()
        time.sleep(2)

    raise RuntimeError(
        f"Timed out waiting for a platform Ray cluster with {expected_gpus} GPU(s) "
        f"after {timeout_s}s. Last Ray error: {last_error}"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner_name = os.environ.get("ALFWORLD_RUNNER")
    if runner_name not in {"run_alfworld_grpo.sh", "run_alfworld_gigpo.sh"}:
        raise RuntimeError("Set ALGORITHM through launch_alfworld.sh")
    launcher = repo_root / "quakecmd_env" / runner_name
    if not (repo_root / "verl").is_dir() or not launcher.is_file():
        raise RuntimeError(f"Incomplete official verl-agent checkout under {repo_root}")

    os.chdir(repo_root)
    os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo_root)],
        check=True,
    )
    if importlib.util.find_spec("deepspeed") is not None:
        print("Removing unused base-image deepspeed for the FSDP + vLLM run.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "deepspeed"], check=True)

    # Quake starts the Ray head and workers on all pods.  Only rank 0 should
    # submit the single veRL driver to that shared Ray cluster.
    if _rank() != 0:
        return

    expected_gpus = int(os.environ["GPU_PER_POD"]) * int(os.environ["TRAINER_NNODES"])
    _wait_for_platform_ray(expected_gpus)
    raise SystemExit(subprocess.call(["bash", str(launcher), "vllm", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
