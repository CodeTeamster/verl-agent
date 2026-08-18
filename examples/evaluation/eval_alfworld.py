"""Evaluate an ALFWorld actor checkpoint with the existing multi-turn rollout.

Example:
    python examples/evaluation/eval_alfworld.py \
        evaluation.checkpoint_path=/path/to/global_step_10_or_hf_model \
        data.val_files=./assets/datasets/alfworld_grpo_train16_val32/text/test.parquet \
        data.val_batch_size=32 \
        actor_rollout_ref.model.path=/path/to/base_model \
        env.env_name=alfworld/AlfredTWEnv \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False

``evaluation.checkpoint_path`` accepts either a verl ``global_step_N`` directory
(not its ``actor`` subdirectory) or a Hugging Face model directory containing
``config.json``. This evaluator deliberately reuses ``TrajectoryCollector`` so
its prompt construction, action parsing, and environment interaction match training.
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.dataset.rl_dataset import collate_fn


def _build_actor_rollout_worker(config):
    """Create only the actor+rollout worker used for inference."""
    strategy = config.actor_rollout_ref.actor.strategy
    if strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        worker_group_cls = RayWorkerGroup
    elif strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise ValueError(f"Unsupported actor strategy: {strategy}")

    pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=1,
        name_prefix="alfworld_eval_",
    )
    actor_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    colocated_cls = create_colocated_worker_cls({"actor_rollout": actor_cls})
    worker_groups = worker_group_cls(
        resource_pool=pool,
        ray_cls_with_init=colocated_cls,
        device_name=config.trainer.device,
    ).spawn(prefix_set={"actor_rollout"})
    actor_rollout_wg = worker_groups["actor_rollout"]
    actor_rollout_wg.init_model()
    return actor_rollout_wg


def _build_validation_envs(config):
    """Create the same normalized validation environment interface used by training."""
    if config.env.env_name != "alfworld/AlfredTWEnv":
        raise ValueError("This evaluator supports env.env_name=alfworld/AlfredTWEnv only.")

    from functools import partial

    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs

    project_root = Path(__file__).resolve().parents[2]
    alf_config_path = project_root / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    if config.data.val_batch_size is None:
        raise ValueError("Set data.val_batch_size explicitly for evaluation.")
    batch_size = config.data.val_batch_size * config.actor_rollout_ref.rollout.val_kwargs.n

    raw_envs = build_alfworld_envs(
        alf_config_path=str(alf_config_path),
        seed=config.env.seed + 1000,
        env_num=batch_size,
        group_n=1,
        is_train=False,
        resources_per_worker=config.env.resources_per_worker,
        env_kwargs={"eval_dataset": config.env.alfworld.eval_dataset},
        envs_per_worker=config.env.alfworld.get("envs_per_worker", 1),
    )
    return AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)


def _make_generation_batch(batch, tokenizer, do_sample, val_repeat):
    """Keep exactly the fields consumed by the existing rollout loop."""
    data = DataProto.from_single_dict(batch)
    data = data.repeat(repeat_times=val_repeat, interleave=True)
    batch_keys = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_keys = ["raw_prompt_ids", "data_source"]
    for key in ("raw_prompt", "tools_kwargs", "env_kwargs", "multi_modal_data"):
        if key in data.non_tensor_batch:
            non_tensor_keys.append(key)

    generation_batch = data.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
    generation_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": do_sample,
        "validate": True,
    }
    return generation_batch


def _trajectory_token_counts(trajectories):
    """Return active prompt and generated token counts for each trajectory."""
    prompt_counts = []
    response_counts = []
    for trajectory in trajectories:
        prompt_tokens = 0
        response_tokens = 0
        for step in trajectory:
            if not bool(np.asarray(step["active_masks"]).item()):
                continue
            response_width = step["responses"].shape[-1]
            attention_mask = step["attention_mask"]
            prompt_tokens += int(attention_mask[:-response_width].sum())
            response_tokens += int(attention_mask[-response_width:].sum())
        prompt_counts.append(prompt_tokens)
        response_counts.append(response_tokens)
    return prompt_counts, response_counts


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


@ray.remote(num_cpus=1)
class EvaluationRunner:
    def run(self, config):
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.trainer.main_ppo import _patch_multiprocess_nfs_temp_cleanup, create_rl_dataset
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        runner_start = time.perf_counter()
        _patch_multiprocess_nfs_temp_cleanup()
        checkpoint_path = config.evaluation.checkpoint_path
        if not checkpoint_path:
            raise ValueError("Set evaluation.checkpoint_path to a verl checkpoint or HF model directory.")
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        actor_checkpoint = checkpoint_path / "actor"
        is_verl_checkpoint = actor_checkpoint.is_dir()
        is_hf_model = (checkpoint_path / "config.json").is_file()
        if not is_verl_checkpoint and not is_hf_model:
            raise ValueError(
                "evaluation.checkpoint_path must be a verl global_step_N directory "
                "containing actor/, or an HF model directory containing config.json."
            )

        # A raw HF model already contains actor weights. A verl checkpoint instead
        # needs the base HF model below to construct its sharded FSDP actor first.
        if is_hf_model:
            config.actor_rollout_ref.model.path = str(checkpoint_path)

        if not config.actor_rollout_ref.model.path or not config.data.val_files:
            raise ValueError("Set actor_rollout_ref.model.path and data.val_files.")

        model_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = hf_tokenizer(model_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(model_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        if len(val_dataset) % config.data.val_batch_size:
            raise ValueError("data.val_batch_size must divide the validation dataset size.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=config.data.get("dataloader_num_workers", 0),
        )

        envs = _build_validation_envs(config)
        actor_rollout_wg = _build_actor_rollout_worker(config)
        if is_verl_checkpoint:
            actor_rollout_wg.load_checkpoint(str(actor_checkpoint), del_local_after_load=False)
        collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        total_rewards = []
        total_lengths = []
        success_values = {}
        batch_seconds = []
        trajectory_latencies = []
        trajectory_prompt_tokens = []
        trajectory_response_tokens = []
        prompt_tokens = 0
        response_tokens = 0
        start = time.perf_counter()
        try:
            from tqdm.auto import tqdm

            progress = tqdm(
                total=len(val_dataset) * config.actor_rollout_ref.rollout.val_kwargs.n,
                desc="ALFWorld evaluation",
                unit="game",
            )
            for batch in val_loader:
                generation_batch = _make_generation_batch(
                    batch,
                    tokenizer,
                    config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    config.actor_rollout_ref.rollout.val_kwargs.n,
                )
                batch_start = time.perf_counter()
                trajectories, rewards, lengths, success, _, _, latencies = collector.vanilla_multi_turn_loop(
                    gen_batch=generation_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                    collect_trajectory_timing=True,
                )
                batch_seconds.append(time.perf_counter() - batch_start)
                batch_prompt_tokens, batch_response_tokens = _trajectory_token_counts(trajectories)
                trajectory_latencies.extend(latencies.tolist())
                trajectory_prompt_tokens.extend(batch_prompt_tokens)
                trajectory_response_tokens.extend(batch_response_tokens)
                prompt_tokens += sum(batch_prompt_tokens)
                response_tokens += sum(batch_response_tokens)
                total_rewards.extend(rewards.tolist())
                total_lengths.extend(lengths.tolist())
                for name, values in success.items():
                    success_values.setdefault(name, []).extend(values.tolist())
                progress.update(len(rewards))
                progress.set_postfix(batch_seconds=f"{batch_seconds[-1]:.1f}")
            progress.close()
        finally:
            wall_seconds = time.perf_counter() - start
            envs.close()

        trajectory_details = [
            {
                "index": index,
                "latency_seconds": latency,
                "prompt_tokens": prompt_count,
                "response_tokens": response_count,
                "episode_length": length,
                "episode_reward": reward,
            }
            for index, (latency, prompt_count, response_count, length, reward) in enumerate(
                zip(trajectory_latencies, trajectory_prompt_tokens, trajectory_response_tokens, total_lengths, total_rewards)
            )
        ]
        startup_seconds = start - runner_start
        trajectory_count = len(total_rewards)
        agent_steps = int(sum(total_lengths))
        total_tokens = prompt_tokens + response_tokens
        metrics = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_format": "verl" if is_verl_checkpoint else "huggingface",
            "trajectories": trajectory_count,
            "agent_steps": agent_steps,
            "episode_reward": _summarize(total_rewards),
            "episode_length": _summarize(total_lengths),
            "success_rate": {name: float(np.mean(values)) for name, values in success_values.items()},
            "wall_seconds": wall_seconds,
            "startup_seconds": startup_seconds,
            "batch_wall_seconds": _summarize(batch_seconds),
            "trajectory_latency_seconds": _summarize(trajectory_latencies),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "prompt_tokens_per_trajectory": _summarize(trajectory_prompt_tokens),
            "response_tokens_per_trajectory": _summarize(trajectory_response_tokens),
            "tokens_per_second": total_tokens / wall_seconds if wall_seconds else 0.0,
            "trajectories_per_second": trajectory_count / wall_seconds if wall_seconds else 0.0,
            "amortized_seconds_per_agent_step": wall_seconds / agent_steps if agent_steps else 0.0,
            "trajectory_details": trajectory_details,
        }
        log_metrics = {key: value for key, value in metrics.items() if key != "trajectory_details"}
        print(json.dumps(log_metrics, indent=2, sort_keys=True), flush=True)
        return metrics


@hydra.main(config_path="config", config_name="eval_alfworld", version_base=None)
def main(config):
    """Start Ray, run one actor-only evaluation, then release local resources."""
    started_ray = not ray.is_initialized()
    if started_ray:
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_runtime_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        runner = EvaluationRunner.remote()
        metrics = ray.get(runner.run.remote(config))
        output_path = config.get("evaluation", {}).get("output_path")
        if output_path:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    finally:
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    main()
