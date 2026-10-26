from time import perf_counter
from typing import NamedTuple

import numpy as np

from .rollout import RolloutState, rollout


class TrainingStep(NamedTuple):
    timesteps: int
    episode_returns: list[float]
    rollout_seconds: float
    update_seconds: float


def learn(env, ppo, total_timesteps, rollout_steps):
    """Train in full rollouts, yielding progress after each update. The timestep budget is a minimum."""
    if total_timesteps < 0:
        raise ValueError('total_timesteps must be nonnegative')
    if rollout_steps <= 0 or rollout_steps % env.num_envs:
        raise ValueError('rollout_steps must be a positive multiple of env.num_envs')

    state = None
    for timesteps in range(0, total_timesteps, rollout_steps):
        if state is None:
            state = RolloutState(env.reset()[0], np.zeros(env.num_envs))
        started = perf_counter()
        batch, returns = rollout(env, ppo.actor_critic, num_steps=rollout_steps, state=state)
        collected = perf_counter()
        ppo.update(batch)
        updated = perf_counter()
        yield TrainingStep(timesteps + batch.rewards.size, returns, collected - started, updated - collected)
