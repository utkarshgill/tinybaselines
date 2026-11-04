import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor

from tinybaselines import GaussianActorCritic, PPO, rollout
from tinybaselines.rollout import RolloutBatch


class CoreTest(unittest.TestCase):
    def test_update_metrics_from_known_ratios(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=2,
                  eps_clip=0.25, batch_size=2, vf_coef=1.0, entropy_coef=0.0)
        for ratios in (np.ones(5), np.array([0.5, 0.75, 1.0, 1.25, 2.0])):
            with self.subTest(ratios=ratios):
                prepared = [Tensor.zeros(5, 3), Tensor.zeros(5, 1), Tensor.zeros(5), Tensor.zeros(5), Tensor.zeros(5)]
                logprobs = Tensor(np.log(ratios).astype(np.float32))
                with patch.object(ppo, 'prepare', return_value=prepared), patch.object(ppo, 'train_step') as step, \
                     patch.object(policy, 'evaluate_actions', return_value=(None, logprobs, None)):
                    metrics = ppo.update(None)
                self.assertEqual(step.call_count, 6)
                self.assertAlmostEqual(metrics['approx_kl'], np.mean(ratios - 1 - np.log(ratios)), places=6)
                self.assertEqual(metrics['clip_fraction'], np.mean(np.abs(ratios - 1) > ppo.eps_clip))

    def test_action_dtype_and_shape_survive_minibatching(self):
        observations = np.zeros((4, 3), np.float32)
        done = np.zeros(4, bool)
        env = SimpleNamespace(num_envs=4, reset=lambda: (observations, {}),
                              step=lambda actions: (observations, np.ones(4), done, done, {}))
        policy = GaussianActorCritic(3, 2, 4, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=4, vf_coef=1.0, entropy_coef=0.0)
        values = Tensor.zeros(8, 1, device=policy.device)
        for actions in (np.arange(4, dtype=np.int32), np.arange(8, dtype=np.float32).reshape(4, 2) / 8):
            with self.subTest(dtype=actions.dtype, shape=actions.shape):
                with patch.object(policy, 'act', return_value=(actions, actions)):
                    batch, _ = rollout(env, policy, num_steps=8)
                self.assertEqual(batch.raw_actions.dtype, actions.dtype)
                np.testing.assert_array_equal(batch.raw_actions, np.stack([actions, actions]))
                with patch.object(policy, 'evaluate_actions', return_value=(values, values.squeeze(-1), None)), \
                     patch.object(ppo, 'train_step') as step, patch('numpy.random.permutation', return_value=np.arange(8)):
                    ppo.update(batch)
                minibatches = np.concatenate([call.args[1].numpy() for call in step.call_args_list])
                self.assertEqual(minibatches.dtype, actions.dtype)
                np.testing.assert_array_equal(minibatches, np.concatenate([actions, actions]))

    def test_ppo_gets_actor_parameters_and_device_from_policy(self):
        model = GaussianActorCritic(3, 1, 4, 1, 1)
        weight = model.actor.layers[0].weight
        policy = SimpleNamespace(actor_parameters=[weight], device=weight.device, critic=model.critic)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=4, vf_coef=1.0, entropy_coef=0.0)
        self.assertEqual(ppo.device, policy.device)
        self.assertIs(ppo.pi_optimizer.params[0], weight)

    def test_rejects_nonpositive_update_settings(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        settings = dict(pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                        eps_clip=0.2, batch_size=4, vf_coef=1.0, entropy_coef=0.0)
        for name in ('K_epochs', 'batch_size'):
            for value in (0, -1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, f'{name} must be positive'):
                    PPO(policy, **(settings | {name: value}))

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
                states, actions, old_logprobs, _, _ = ppo.prepare(batch)
                old_logprobs = old_logprobs.numpy().copy()
                metrics = ppo.update(batch)
                new_logprobs = policy.evaluate_actions(states, actions)[1].numpy()
                log_ratio = new_logprobs - old_logprobs
                self.assertAlmostEqual(metrics['approx_kl'], np.mean(np.expm1(log_ratio) - log_ratio), places=6)
                self.assertEqual(metrics['clip_fraction'], np.mean(np.abs(np.exp(log_ratio) - 1) > ppo.eps_clip))
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
