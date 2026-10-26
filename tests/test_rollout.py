import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import lander
import reference_lander


class RolloutTest(unittest.TestCase):
    def test_episode_endings_remain_distinct(self):
        terminated = np.array([True, False, True, False])
        truncated = np.array([False, True, True, False])
        states = np.zeros((4, 8), dtype=np.float32)
        final_obs = np.full((4, 8), 7, dtype=np.float32)
        actions = np.zeros((4, 2), dtype=np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        env = SimpleNamespace(num_envs=4, reset=lambda: (states, {}),
                              step=lambda actions: (states, np.ones(4), terminated, truncated, {'final_obs': final_obs}))
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                result = trainer.rollout(env, policy, num_steps=4)
                np.testing.assert_array_equal(result[3][0], terminated)
                np.testing.assert_array_equal(result[4][0], truncated)
                np.testing.assert_array_equal(result[5][0, :3], final_obs[:3])
                np.testing.assert_array_equal(result[5][0, 3], states[3])
                self.assertEqual(len(result[-1]), 3)

    def test_each_step_is_a_transition(self):
        make = lander.gym.make
        actions = np.zeros((4, 2), dtype=np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                with patch.object(trainer.gym, 'make', side_effect=lambda *a, **kw: make(*a, max_episode_steps=1, **kw)):
                    env = trainer.make_env(4, asynchronous=False)
                try:
                    _, _, _, terminated, truncated, _, episodes = trainer.rollout(env, policy, num_steps=8)
                    np.testing.assert_array_equal(terminated, np.zeros((2, 4), bool))
                    np.testing.assert_array_equal(truncated, np.ones((2, 4), bool))
                    self.assertEqual(len(episodes), 8)
                finally:
                    env.close()

    def test_next_observations_are_snapshots(self):
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                states = np.zeros((4, 8), dtype=np.float32)

                def step(actions):
                    states[:] += 1
                    return states, np.ones(4), np.zeros(4, bool), np.zeros(4, bool), {}

                env = SimpleNamespace(num_envs=4, reset=lambda: (states, {}), step=step)
                actions = np.zeros((4, 2), dtype=np.float32)
                policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
                obs, _, _, _, _, next_obs, _ = trainer.rollout(env, policy, num_steps=8)
                np.testing.assert_array_equal(obs[:, 0, 0], [0, 1])
                np.testing.assert_array_equal(next_obs[:, 0, 0], [1, 2])
                states[:] = 99
                np.testing.assert_array_equal(next_obs[:, 0, 0], [1, 2])

    def test_bootstrap_values(self):
        cases = (
            ([[0, 1]], [[2, 3]], [[0, 1]], [10, 20], [9.9, 1]),
            ([[0], [0]], [[2], [3]], [[0], [0]], [10], [9.45945, 9.9]),
        )
        for trainer in (lander, reference_lander):
            for rewards, values, done, last_values, expected in cases:
                with self.subTest(trainer=trainer.__name__, expected=expected):
                    ppo = trainer.PPO.__new__(trainer.PPO)
                    ppo.gamma, ppo.lamda = 0.99, 0.95
                    arrays = [np.asarray(x, dtype=np.float32) for x in (rewards, values, done, last_values)]
                    if trainer is reference_lander:
                        arrays = [trainer.torch.from_numpy(x) for x in arrays]
                    advantages, returns = ppo.compute_advantages(*arrays)
                    np.testing.assert_allclose(returns, expected, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(advantages, [1, -1], rtol=1e-5, atol=1e-5)
