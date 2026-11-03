from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np

from tinybaselines import CategoricalActorCritic, PPO, evaluate, learn


class CartPoleTest(unittest.TestCase):
    def test_categorical_policy_trains_with_ppo(self):
        policy = CategoricalActorCritic(4, 2, 16, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=2,
                  eps_clip=0.2, batch_size=64, vf_coef=0.5, entropy_coef=0.01)
        weight = policy.actor.layers[-1].weight
        before = weight.numpy().copy()
        with closing(gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 4,
                                              autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)) as env:
            actions, _ = policy.act(env.reset()[0])
            self.assertTrue(env.action_space.contains(actions))
            steps = list(learn(env, ppo, total_timesteps=384, rollout_steps=128))
            self.assertEqual(steps[-1].timesteps, 384)
            self.assertFalse(np.array_equal(weight.numpy(), before))
            observations, _ = env.reset()
            with TemporaryDirectory() as directory:
                path = Path(directory) / 'ppo.safetensors'
                ppo.save(path)
                restored = PPO.load(path)
                self.assertIsInstance(restored.actor_critic, CategoricalActorCritic)
                np.testing.assert_array_equal(restored.actor_critic.act(observations, deterministic=True)[0],
                                              policy.act(observations, deterministic=True)[0])
                state = restored.state_dict()
                self.assertEqual(state.keys(), ppo.state_dict().keys())
                for name, tensor in ppo.state_dict().items():
                    np.testing.assert_array_equal(state[name].numpy(), tensor.numpy())
                self.assertEqual(len(evaluate(env, restored.actor_critic, num_episodes=8)), 8)
                self.assertEqual(next(learn(env, restored, total_timesteps=128, rollout_steps=128)).timesteps, 128)
