import unittest

import numpy as np
from tinygrad import Tensor

from tinybaselines import CategoricalActorCritic, GaussianActorCritic, PPO, learn


class IdentityEnv:
    num_envs = 4

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.observations = self.rng.uniform(-0.75, 0.75, (self.num_envs, 1)).astype(np.float32)
        return self.observations, {}

    def step(self, actions):
        rewards = -np.square(actions[:, 0] - self.observations[:, 0])
        observations, _ = self.reset()
        return observations, rewards, np.ones(self.num_envs, bool), np.zeros(self.num_envs, bool), {}


def identity_errors(seed):
    Tensor.manual_seed(seed)
    np.random.seed(seed)
    policy = GaussianActorCritic(1, 1, 16, 1, 1)
    ppo = PPO(policy, pi_lr=0.01, vf_lr=0.01, gamma=0.99, lamda=0.95, K_epochs=4,
              eps_clip=0.2, batch_size=256, vf_coef=0.5, entropy_coef=0.0)
    observations = np.linspace(-0.75, 0.75, 65, dtype=np.float32)[:, None]

    def error():
        actions, _ = policy.act(observations, deterministic=True)
        return np.square(actions - observations).mean()

    before = error()
    for _ in learn(IdentityEnv(seed), ppo, total_timesteps=2048, rollout_steps=256):
        pass
    return before, error(), np.square(observations).mean()


class DiscreteIdentityEnv:
    num_envs = 4
    observations = np.eye(4, dtype=np.float32)

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.targets = self.rng.integers(4, size=self.num_envs)
        return self.observations[self.targets], {}

    def step(self, actions):
        rewards = (actions == self.targets).astype(np.float32)
        observations, _ = self.reset()
        return observations, rewards, np.ones(self.num_envs, bool), np.zeros(self.num_envs, bool), {}


def discrete_identity_scores(seed):
    Tensor.manual_seed(seed)
    np.random.seed(seed)
    env = DiscreteIdentityEnv(seed)
    policy = CategoricalActorCritic(4, 4, 16, 1, 1)
    ppo = PPO(policy, pi_lr=0.02, vf_lr=0.01, gamma=0.99, lamda=0.95, K_epochs=4,
              eps_clip=0.2, batch_size=64, vf_coef=0.5, entropy_coef=0.0)
    states = policy.preprocess(env.observations)
    targets = Tensor(np.arange(4, dtype=np.int32), device=policy.device)

    def score():
        _, log_probs, _ = policy.evaluate_actions(states, targets)
        return log_probs.exp().mean().item()

    before = score()
    for _ in learn(env, ppo, total_timesteps=1024, rollout_steps=128):
        pass
    return before, score()


class LearningTest(unittest.TestCase):
    def test_learns_identity(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        before, after, zero = identity_errors(seed=0)
        self.assertLess(after, 0.25 * min(before, zero), (before, after, zero))

    def test_learns_discrete_identity(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        before, after = discrete_identity_scores(seed=0)
        self.assertGreater(after, max(before, 0.9), (before, after))
