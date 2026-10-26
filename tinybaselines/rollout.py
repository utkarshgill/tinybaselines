from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class RolloutState:
    observations: np.ndarray
    episode_returns: np.ndarray


class RolloutBatch(NamedTuple):
    observations: np.ndarray
    raw_actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_observations: np.ndarray


def track_episode_returns(done_mask, ep_returns, ep_rets, remaining=None):
    for idx in np.where(done_mask)[0]:
        if remaining is None or remaining[idx] > 0:
            ep_returns.append(ep_rets[idx])
            if remaining is not None:
                remaining[idx] -= 1
        ep_rets[idx] = 0.0


def rollout(env, actor_critic, num_steps=None, num_episodes=None, deterministic=False, state=None):
    assert (num_steps is None) != (num_episodes is None), 'Specify exactly one: num_steps or num_episodes'
    if num_episodes is not None and num_episodes <= 0:
        raise ValueError('num_episodes must be positive')

    N = env.num_envs
    if state is None:
        state = RolloutState(env.reset()[0], np.zeros(N))
    states = state.observations
    ep_returns, ep_rets = [], state.episode_returns

    collect = num_steps is not None
    remaining = None if collect else (num_episodes + np.arange(N)) // N
    if collect:
        T = num_steps // N
        obs = np.empty((T, *states.shape), dtype=np.float32)
        next_obs = np.empty_like(obs)
        rew = np.empty((T, N), dtype=np.float32)
        terminations = np.empty((T, N), dtype=bool)
        truncations = np.empty((T, N), dtype=bool)

    t = 0
    while True:
        actions, raw_actions = actor_critic.act(states, deterministic=deterministic)
        if collect:
            if t == 0:
                raw_act = np.empty((T, *raw_actions.shape), dtype=np.float32)
            obs[t], raw_act[t] = states, raw_actions

        states, rewards, terminated, truncated, infos = env.step(actions)
        d = np.logical_or(terminated, truncated)
        if collect:
            rew[t], terminations[t], truncations[t] = rewards, terminated, truncated
            next_obs[t] = states
            final_obs = infos.get('final_obs')
            for idx in np.flatnonzero(d):
                if final_obs is not None and final_obs[idx] is not None:
                    next_obs[t, idx] = final_obs[idx]
                elif truncated[idx] and not terminated[idx]:
                    raise ValueError('truncated transitions require final_obs')

        ep_rets += rewards
        track_episode_returns(d, ep_returns, ep_rets, remaining)
        t += 1

        if (collect and t >= T) or (not collect and not remaining.any()):
            break

    state.observations = states
    return (RolloutBatch(obs, raw_act, rew, terminations, truncations, next_obs), ep_returns) if collect else ep_returns
