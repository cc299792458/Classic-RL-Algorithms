import numpy as np
import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import List, Callable


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize the function. This is required for multiprocessing.
    """
    def __init__(self, x):
        self.x = x

    def __call__(self):
        import cloudpickle
        return cloudpickle.loads(self.x)()

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            remote.send(env.step(data))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class SubprocVecEnv:
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        env_fns: List of environment functions to be run in parallel.
        """
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for process in self.processes:
            process.start()
        for work_remote in self.work_remotes:
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, terminateds, truncateds, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(terminateds), np.stack(truncateds), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()

def make_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4
    env_fns = [make_env('Blackjack-v1', seed=i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    obs = env.reset()
    print("Initial observations:", obs)

    for _ in range(1000):
        actions = [env.action_space.sample() for _ in range(num_envs)]
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

    env.close()
