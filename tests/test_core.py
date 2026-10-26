import subprocess
import sys
import unittest

import numpy as np

from tinybaselines import GaussianActorCritic, PPO
from tinybaselines.rollout import RolloutBatch


class CoreTest(unittest.TestCase):
    def test_import_without_example_dependencies(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import tinybaselines; '
                        'assert not {"examples", "gymnasium", "matplotlib", "torch", "tqdm"}.intersection(sys.modules)'],
                       check=True)

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
