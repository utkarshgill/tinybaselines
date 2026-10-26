from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tinybaselines import GaussianActorCritic, PPO, learn


class TrainingTest(unittest.TestCase):
    def setUp(self):
        self.observations = np.zeros((2, 3), np.float32)
        actions = np.zeros((2, 1), np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        self.ppo = SimpleNamespace(actor_critic=policy, update=Mock())

        def step(actions):
            self.observations[:] += 1
            terminated = self.observations[:, 0] == 3
            self.observations[terminated] = 0
            return self.observations, np.ones(2), terminated, np.zeros(2, bool), {}

        self.env = SimpleNamespace(num_envs=2, reset=Mock(return_value=(self.observations, {})),
                                   step=Mock(side_effect=step), close=Mock())

    def test_counts_full_rollouts_and_preserves_episodes(self):
        with patch('tinybaselines.training.perf_counter', side_effect=range(9)):
            steps = list(learn(self.env, self.ppo, total_timesteps=9, rollout_steps=4))
        self.assertEqual([step.timesteps for step in steps], [4, 8, 12])
        self.assertEqual([step.episode_returns for step in steps], [[], [3, 3], [3, 3]])
        self.assertEqual([(step.rollout_seconds, step.update_seconds) for step in steps], [(1, 1)] * 3)
        batches = [call.args[0] for call in self.ppo.update.call_args_list]
        np.testing.assert_array_equal([batch.observations[:, 0, 0] for batch in batches], [[0, 1], [2, 0], [1, 2]])
        self.env.reset.assert_called_once()
        self.assertEqual(self.env.step.call_count, 6)
        self.env.close.assert_not_called()

    def test_rejects_invalid_budgets_before_reset(self):
        for total, rollout_steps in ((-1, 4), (8, 0), (8, -4), (8, 3)):
            with self.subTest(total=total, rollout_steps=rollout_steps):
                with self.assertRaises(ValueError):
                    list(learn(self.env, self.ppo, total_timesteps=total, rollout_steps=rollout_steps))
        self.env.reset.assert_not_called()
        self.ppo.update.assert_not_called()

    def test_zero_budget_does_no_work(self):
        self.assertEqual(list(learn(self.env, self.ppo, total_timesteps=0, rollout_steps=4)), [])
        self.env.reset.assert_not_called()
        self.env.step.assert_not_called()
        self.ppo.update.assert_not_called()

    def test_caller_can_stop_after_an_update(self):
        training = learn(self.env, self.ppo, total_timesteps=100, rollout_steps=4)
        self.assertEqual(next(training).timesteps, 4)
        self.ppo.update.assert_called_once()
        training.close()
        self.assertEqual(self.env.step.call_count, 2)
        self.env.close.assert_not_called()

    def test_update_failure_reaches_caller(self):
        self.ppo.update.side_effect = RuntimeError('failed update')
        with self.assertRaisesRegex(RuntimeError, 'failed update'):
            next(learn(self.env, self.ppo, total_timesteps=4, rollout_steps=4))
        self.env.close.assert_not_called()

    def test_trains_a_policy(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=4, vf_coef=1.0, entropy_coef=0.001)
        before = policy.critic.layers[-1].bias.numpy().copy()
        steps = list(learn(self.env, ppo, total_timesteps=12, rollout_steps=4))
        self.assertEqual(steps[-1].timesteps, 12)
        self.assertFalse(np.array_equal(before, policy.critic.layers[-1].bias.numpy()))
