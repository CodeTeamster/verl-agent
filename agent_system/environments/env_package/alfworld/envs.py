import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
import sys

from alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
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

def worker_func(remote, config, seeds, env_type, train_eval, base_env):
    """Run one or more independent ALFWorld environments in one subprocess."""
    envs = []
    for index, seed in enumerate(seeds):
        env = base_env.init_env(batch_size=1)
        env.seed(seed)
        envs.append(env)

        # TextWorld creates a distinct object for init_env(), whereas some
        # backends reuse the base object. Keep every slot independent.
        if env is base_env and index + 1 < len(seeds):
            base_env = get_environment(env_type)(config, train_eval=train_eval)

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if len(data) != len(envs):
                raise ValueError(f"Expected {len(envs)} actions, got {len(data)}")
            results = []
            for env, action in zip(envs, data):
                obs, scores, dones, infos = env.step([action])
                infos['observation_text'] = obs
                results.append((obs, scores, dones, infos))
            remote.send(results)

        elif cmd == 'reset':
            results = []
            for env in envs:
                obs, infos = env.reset()
                infos['observation_text'] = obs
                results.append((obs, infos))
            remote.send(results)

        elif cmd == 'getobs':
            remote.send([get_obs_image(env).cpu() for env in envs])

        elif cmd == 'close':
            for env in envs:
                close = getattr(env, 'close', None)
                if callable(close):
                    close()
            remote.close()
            break

        else:
            raise NotImplementedError("Unknown command: {}".format(cmd))

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, envs_per_worker=1):
        super().__init__()
        if not isinstance(envs_per_worker, int) or isinstance(envs_per_worker, bool) or envs_per_worker < 1:
            raise ValueError(f"envs_per_worker must be a positive integer, got {envs_per_worker!r}")
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        train_eval = 'train' if is_train else 'eval_in_distribution'
        base_env = get_environment(env_type)(config, train_eval=train_eval)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.envs_per_worker = envs_per_worker

        self.parent_remotes = []
        self.workers = []
        self.worker_sizes = []

        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork')

        all_seeds = [seed + (i // self.group_n) for i in range(self.num_processes)]
        for start in range(0, self.num_processes, self.envs_per_worker):
            worker_seeds = all_seeds[start:start + self.envs_per_worker]
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=worker_func,
                args=(child_remote, config, worker_seeds, env_type, train_eval, base_env)
            )
            worker.daemon = True
            worker.start()

            child_remote.close()

            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)
            self.worker_sizes.append(len(worker_seeds))

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        offset = 0
        for remote, worker_size in zip(self.parent_remotes, self.worker_sizes):
            remote.send(('step', actions[offset:offset + worker_size]))
            offset += worker_size

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        worker_results = [remote.recv() for remote in self.parent_remotes]
        for i, (obs, scores, dones, info) in enumerate(
                result for results in worker_results for result in results):
            for k in info.keys():
                info[k] = info[k][0]

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
        Send the reset command to all subprocesses at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        for remote in self.parent_remotes:
            remote.send(('reset', None))

        worker_results = [remote.recv() for remote in self.parent_remotes]
        for i, (obs, info) in enumerate(result for results in worker_results for result in results):
            for k in info.keys():
                info[k] = info[k][0] 
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
        Ask each subprocess to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        for remote in self.parent_remotes:
            remote.send(('getobs', None))

        worker_image_batches = [remote.recv() for remote in self.parent_remotes]
        return [image for image_batch in worker_image_batches for image in image_batch]

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all subprocesses
        """
        for remote in self.parent_remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, envs_per_worker=1):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, envs_per_worker)
