import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tinybaselines import GaussianActorCritic, PPO, rollout
from tinybaselines.rollout import RolloutBatch


class CoreTest(unittest.TestCase):
    def test_terminals_do_not_require_final_observations(self):
        observations = np.zeros((2, 3), np.float32)
        reset_observations = np.full((2, 3), 100, np.float32)
        actions = np.zeros((2, 1), np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        terminated, truncated = np.ones(2, bool), np.array([False, True])
        env = SimpleNamespace(num_envs=2, reset=lambda: (observations, {}),
                              step=lambda actions: (reset_observations, np.array([1, 2]), terminated, truncated, {}))
        batch, episodes = rollout(env, policy, num_steps=2)
        np.testing.assert_array_equal(episodes, [1, 2])
        np.testing.assert_array_equal(batch.next_observations[0], reset_observations)
        ppo = PPO.__new__(PPO)
        ppo.gamma, ppo.lamda = 0.99, 0.95
        _, returns = ppo.compute_advantages(batch.rewards, np.zeros((1, 2)), batch.terminated,
                                            batch.truncated, np.full((1, 2), 100))
        np.testing.assert_array_equal(returns, [1, 2])

    def test_truncations_require_final_observations(self):
        observations = np.zeros((1, 3), np.float32)
        actions = np.zeros((1, 1), np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        for infos in ({}, {'final_obs': None}, {'final_obs': [None]}):
            with self.subTest(infos=infos):
                env = SimpleNamespace(num_envs=1, reset=lambda: (observations, {}),
                                      step=lambda actions: (observations, np.ones(1), np.zeros(1, bool), np.ones(1, bool), infos))
                with self.assertRaisesRegex(ValueError, 'truncated transitions require final_obs'):
                    rollout(env, policy, num_steps=1)

    def test_rollout_uses_observation_and_action_shapes(self):
        observations = np.zeros((2, 3), np.float32)
        actions = np.zeros((2, 1), np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        env = SimpleNamespace(num_envs=2, reset=lambda: (observations, {}),
                              step=lambda actions: (observations, np.ones(2), np.zeros(2, bool), np.zeros(2, bool), {}))
        batch, episodes = rollout(env, policy, num_steps=6)
        self.assertEqual(batch.observations.shape, (3, 2, 3))
        self.assertEqual(batch.raw_actions.shape, (3, 2, 1))
        self.assertEqual(episodes, [])

    def test_import_without_example_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-I', '-c',
                            'import sys; import tinybaselines; '
                            'assert not {"examples", "gymnasium", "matplotlib", "torch", "tqdm"}.intersection(sys.modules)'],
                           cwd=directory, check=True)

    def test_partial_minibatches(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        observations = np.linspace(-0.5, 0.5, 36, dtype=np.float32).reshape(3, 4, 3)
        batch = RolloutBatch(observations, np.zeros((3, 4, 1), np.float32), np.ones((3, 4), np.float32),
                             np.zeros((3, 4), bool), np.zeros((3, 4), bool), observations + 0.1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=2,
                  eps_clip=0.2, batch_size=8, vf_coef=1.0, entropy_coef=0.001)
        with patch.object(ppo, 'train_step', wraps=ppo.train_step) as step:
            for _ in range(3):
                ppo.update(batch)
        self.assertEqual([call.args[0].shape[0] for call in step.call_args_list], [8, 4] * 6)
        for tensor in ppo.state_dict().values():
            self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_policy_and_ppo_use_batch_dimensions(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        observations = np.array([[0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=np.float32)
        np.testing.assert_array_equal(policy.preprocess(observations).numpy(), observations)
        batch = RolloutBatch(observations[None], np.zeros((1, 2, 1), np.float32), np.ones((1, 2), np.float32),
                             np.zeros((1, 2), bool), np.zeros((1, 2), bool), (observations + 0.1)[None])
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=2, vf_coef=1.0, entropy_coef=0.001)
        before = policy.critic.layers[-1].bias.numpy().copy()
        for _ in range(3):
            ppo.update(batch)
        self.assertFalse(np.array_equal(before, policy.critic.layers[-1].bias.numpy()))
