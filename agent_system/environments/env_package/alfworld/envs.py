# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment


def _patch_fast_downward_temp_cleanup():
    """Avoid ALFWorld/TextWorld reset failures on NAS/NFS temp dirs.

    fast_downward loads a copied shared library from TemporaryDirectory. On
    network filesystems, unloading can leave .nfs* files briefly, making rmtree
    raise Directory-not-empty even though the library loaded successfully.
    """
    try:
        import fast_downward.interface as fast_downward_interface
    except Exception:
        return

    temporary_directory = fast_downward_interface.tempfile.TemporaryDirectory
    if getattr(temporary_directory, "_verl_agent_ignore_cleanup_errors", False):
        return

    class IgnoreCleanupErrorsTemporaryDirectory(temporary_directory):
        _verl_agent_ignore_cleanup_errors = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)

    fast_downward_interface.tempfile.TemporaryDirectory = IgnoreCleanupErrorsTemporaryDirectory


_patch_fast_downward_temp_cleanup()

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one or more independent environment instances.
    """

    def __init__(self, config, seeds, env_type, train_eval, base_env):
        self.envs = []
        for index, seed in enumerate(seeds):
            env = base_env.init_env(batch_size=1)
            env.seed(seed)
            self.envs.append(env)

            # TextWorld returns a separate environment and can reuse the base
            # object without rescanning all game files. Other backends may
            # mutate and return the base object itself, in which case the next
            # slot needs a fresh base instance.
            if env is base_env and index + 1 < len(seeds):
                base_env = get_environment(env_type)(config, train_eval=train_eval)

    def step(self, actions):
        """Execute one action for every environment managed by this actor."""
        if len(actions) != len(self.envs):
            raise ValueError(f"Expected {len(self.envs)} actions, got {len(actions)}")

        results = []
        for env, action in zip(self.envs, actions):
            obs, scores, dones, infos = env.step([action])
            infos['observation_text'] = obs
            results.append((obs, scores, dones, infos))
        return results

    def reset(self):
        """Reset every environment managed by this actor."""
        results = []
        for env in self.envs:
            obs, infos = env.reset()
            infos['observation_text'] = obs
            results.append((obs, infos))
        return results

    def getobs(self):
        """Get the current observation image for every managed environment."""
        return [get_obs_image(env).cpu() for env in self.envs]

    def close(self):
        """Release resources owned by all underlying ALFWorld environments."""
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self.envs = []
        return True

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None, envs_per_worker=1):
        super().__init__()

        if not isinstance(envs_per_worker, int) or isinstance(envs_per_worker, bool) or envs_per_worker < 1:
            raise ValueError(f"envs_per_worker must be a positive integer, got {envs_per_worker!r}")

        env_kwargs = env_kwargs or {}
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        train_eval = 'train' if is_train else eval_dataset
        base_env = get_environment(env_type)(config, train_eval=train_eval)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_envs = env_num * group_n
        self.envs_per_worker = envs_per_worker
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        # Task-event collection is unnecessary for these numerous environment actors
        # and can crash Ray 2.43 while FlushEvents() runs at shutdown.
        actor_options = dict(resources_per_worker)
        actor_options.setdefault("enable_task_events", False)
        env_worker = ray.remote(**actor_options)(AlfworldWorker)
        self.workers = []
        self.worker_sizes = []
        all_seeds = [seed + (i // self.group_n) for i in range(self.num_envs)]
        for start in range(0, self.num_envs, self.envs_per_worker):
            worker_seeds = all_seeds[start:start + self.envs_per_worker]
            # resources_per_worker historically described one environment.
            # Scale CPU/GPU claims so actor packing does not silently change
            # the total resources advertised to Ray.
            scaled_options = {}
            for resource_name in ("num_cpus", "num_gpus"):
                if resource_name in actor_options:
                    scaled_options[resource_name] = actor_options[resource_name] * len(worker_seeds)
            worker = env_worker.options(**scaled_options).remote(
                config,
                worker_seeds,
                env_type,
                train_eval,
                base_env,
            )
            self.workers.append(worker)
            self.worker_sizes.append(len(worker_seeds))

        self.prev_admissible_commands = [None for _ in range(self.num_envs)]

    def step(self, actions):
        assert len(actions) == self.num_envs, \
            "The num of actions must be equal to the num of environments"

        # Send step commands to all workers
        futures = []
        offset = 0
        for worker, worker_size in zip(self.workers, self.worker_sizes):
            action_chunk = actions[offset:offset + worker_size]
            future = worker.step.remote(action_chunk)
            futures.append(future)
            offset += worker_size

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        worker_results = ray.get(futures)
        results = [result for worker_result in worker_results for result in worker_result]
        for i, (obs, scores, dones, info) in enumerate(results):
            info = {key: value[0] for key, value in info.items()}

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        worker_results = ray.get(futures)
        results = [result for worker_result in worker_results for result in worker_result]
        for i, (obs, info) in enumerate(results):
            info = {key: value[0] for key, value in info.items()}
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        worker_images = ray.get(futures)
        return [image for images in worker_images for image in images]

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self, timeout_s=30):
        """Close all workers gracefully, force-killing only on timeout."""
        if not self.workers:
            return

        workers = self.workers
        self.workers = []

        close_refs = [worker.close.remote() for worker in workers]
        _, pending_close_refs = ray.wait(
            close_refs, num_returns=len(close_refs), timeout=timeout_s
        )
        if pending_close_refs:
            print(
                f"Warning: {len(pending_close_refs)} ALFWorld workers did not "
                "finish environment cleanup before timeout."
            )

        terminate_refs = [worker.__ray_terminate__.remote() for worker in workers]
        _, pending_terminate_refs = ray.wait(
            terminate_refs, num_returns=len(terminate_refs), timeout=timeout_s
        )
        if pending_terminate_refs:
            pending_ids = {ref.hex() for ref in pending_terminate_refs}
            for worker, terminate_ref in zip(workers, terminate_refs):
                if terminate_ref.hex() in pending_ids:
                    ray.kill(worker, no_restart=True)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None, envs_per_worker=1):
    return AlfworldEnvs(
        alf_config_path,
        seed,
        env_num,
        group_n,
        resources_per_worker,
        is_train,
        env_kwargs,
        envs_per_worker,
    )
