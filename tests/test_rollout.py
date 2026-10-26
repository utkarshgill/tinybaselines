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
