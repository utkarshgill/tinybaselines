from types import SimpleNamespace
import unittest

import numpy as np

from examples.pendulum.env import make_env
from tinybaselines import GaussianActorCritic, PPO, learn, rollout


class PendulumTest(unittest.TestCase):
    def test_normalized_actions_cover_torque_range(self):
        env = make_env(4)
        try:
            np.testing.assert_array_equal(env.single_action_space.low, [-1])
            np.testing.assert_array_equal(env.single_action_space.high, [1])
            self.assertEqual(env.single_observation_space.shape, (3,))
            for action in (-1.0, 0.0, 1.0):
                np.testing.assert_array_equal(env.envs[0].action(np.array([action], np.float32)), [2 * action])
        finally:
            env.close()

    def test_time_limits_keep_final_observations(self):
        env = make_env(4, max_episode_steps=1)
        actions = np.zeros((4, 1), np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        try:
            batch, returns = rollout(env, policy, num_steps=8)
            self.assertEqual(batch.observations.shape, (2, 4, 3))
            self.assertEqual(batch.raw_actions.shape, (2, 4, 1))
            np.testing.assert_array_equal(batch.terminated, np.zeros((2, 4), bool))
            np.testing.assert_array_equal(batch.truncated, np.ones((2, 4), bool))
            self.assertTrue(np.isfinite(batch.next_observations).all())
            self.assertEqual(len(returns), 8)
        finally:
            env.close()

    def test_updates_with_four_environments(self):
        env = make_env(4, max_episode_steps=1)
        policy = GaussianActorCritic(3, 1, 8, 1, 1, observation_scale=[1, 1, 0.125])
        ppo = PPO(policy, pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=4, vf_coef=0.5, entropy_coef=0.0)
        before = policy.critic.layers[-1].bias.numpy().copy()
        try:
            steps = list(learn(env, ppo, total_timesteps=24, rollout_steps=8))
            self.assertEqual([step.timesteps for step in steps], [8, 16, 24])
            self.assertFalse(np.array_equal(policy.critic.layers[-1].bias.numpy(), before))
        finally:
            env.close()
