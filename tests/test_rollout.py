import unittest
from types import SimpleNamespace

import numpy as np

import lander
import reference_lander


class RolloutTest(unittest.TestCase):
    def test_final_observation_is_a_snapshot(self):
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                states = np.zeros((4, 8), dtype=np.float32)

                def step(actions):
                    states[:] += 1
                    return states, np.ones(4), np.zeros(4, bool), np.zeros(4, bool), {}

                env = SimpleNamespace(num_envs=4, reset=lambda: (states, {}), step=step)
                actions = np.zeros((4, 2), dtype=np.float32)
                policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
                obs, _, _, _, last_obs, _ = trainer.rollout(env, policy, num_steps=8)
                np.testing.assert_array_equal(obs[:, 0, 0], [0, 1])
                np.testing.assert_array_equal(last_obs, np.full((4, 8), 2))
                states[:] = 99
                np.testing.assert_array_equal(last_obs, np.full((4, 8), 2))

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
