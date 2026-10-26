import gymnasium as gym
import numpy as np


def make_env(num_envs=4, **kwargs):
    return gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RescaleAction(gym.make('Pendulum-v1', **kwargs), min_action=np.float32(-1), max_action=np.float32(1))
         for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
