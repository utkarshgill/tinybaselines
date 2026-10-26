import gymnasium as gym


def make_env(num_envs=4, **kwargs):
    return gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RescaleAction(gym.make('Pendulum-v1', **kwargs), min_action=-1.0, max_action=1.0)
         for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
