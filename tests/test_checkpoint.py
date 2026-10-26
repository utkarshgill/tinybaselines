import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.lunar_lander import train as lander, reference as reference_lander
from tinybaselines import GaussianActorCritic, PPO, RolloutBatch


class CheckpointTest(unittest.TestCase):
    def make_ppo(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1, observation_scale=[2, 3, 4])
        return PPO(policy, pi_lr=1e-3, vf_lr=2e-3, gamma=0.97, lamda=0.89, K_epochs=2,
                   eps_clip=0.17, batch_size=4, vf_coef=0.8, entropy_coef=0.03)

    def test_ppo_state_contains_policy_and_adam_buffers(self):
        ppo = self.make_ppo()
        ppo.gamma = 0.9
        ppo.pi_optimizer.lr.assign(lander.Tensor([0.005])).realize()
        self.assertEqual(ppo.config['gamma'], 0.9)
        self.assertAlmostEqual(ppo.config['pi_lr'], 0.005)
        state = ppo.state_dict()
        for name, tensor in ppo.actor_critic.state_dict().items():
            self.assertIs(state['policy.' + name], tensor)
        for name in ('pi_optimizer', 'vf_optimizer'):
            optimizer = getattr(ppo, name)
            for key in ('lr', 'b1_t', 'b2_t'):
                self.assertIs(state[name + '.' + key], getattr(optimizer, key))
            for key in ('m', 'v'):
                for index, tensor in enumerate(getattr(optimizer, key)):
                    self.assertIs(state[f'{name}.{key}.{index}'], tensor)
        self.assertFalse(any('.params.' in name for name in state))

    def test_ppo_checkpoint_continues_updates(self):
        observations = np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(2, 2, 3)
        batch = RolloutBatch(observations, np.zeros((2, 2, 1), np.float32), np.array([[1, 2], [3, 4]], np.float32),
                             np.zeros((2, 2), bool), np.zeros((2, 2), bool), observations + 0.2)
        for updates in (0, 4):
            with self.subTest(updates=updates):
                ppo = self.make_ppo()
                ppo.gamma = 0.9
                ppo.pi_optimizer.lr.assign(lander.Tensor([0.005])).realize()
                inputs = [tensor.contiguous().realize() for tensor in ppo.prepare(batch)]
                for _ in range(updates):
                    ppo.train_step(*inputs)
                for _ in range(3):
                    expected_actions = ppo.actor_critic.act(observations[0], deterministic=True)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'ppo.safetensors'
                    ppo.save(path)
                    restored = PPO.load(path)
                self.assertEqual(restored.config, ppo.config)
                self.assertEqual(restored.actor_critic.config, ppo.actor_critic.config)
                self.assertIsNone(restored._train_step.captured)
                self.assertIsNone(restored.actor_critic.actor_jit.captured)
                self.assertEqual(restored.state_dict().keys(), ppo.state_dict().keys())
                for name, tensor in ppo.state_dict().items():
                    np.testing.assert_array_equal(restored.state_dict()[name].numpy(), tensor.numpy())
                for actual, expected in zip(restored.actor_critic.act(observations[0], deterministic=True), expected_actions):
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                for actual, expected in zip(restored.prepare(batch), ppo.prepare(batch)):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
                for _ in range(4):
                    expected = ppo.train_step(*inputs)
                    actual = restored.train_step(*inputs)
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
                    for name, tensor in ppo.state_dict().items():
                        np.testing.assert_allclose(restored.state_dict()[name].numpy(), tensor.numpy(), rtol=1e-5, atol=1e-6)

    def test_can_replace_a_checkpoint_after_loading_it(self):
        ppo = self.make_ppo()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ppo.safetensors'
            ppo.save(path)
            first = PPO.load(path)
            before = first.actor_critic.log_std.numpy().copy()
            ppo.actor_critic.log_std.assign(lander.Tensor([0.25])).realize()
            ppo.save(path)
            second = PPO.load(path)
            np.testing.assert_array_equal(first.actor_critic.log_std.numpy(), before)
            np.testing.assert_array_equal(second.actor_critic.log_std.numpy(), [0.25])

    def test_failed_save_preserves_previous_checkpoint(self):
        ppo = self.make_ppo()

        def failed_save(state, path, metadata):
            Path(path).write_bytes(b'partial checkpoint')
            raise OSError('failed write')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ppo.safetensors'
            ppo.save(path)
            original = path.read_bytes()
            with patch.object(lander.nn.state, 'safe_save', side_effect=failed_save):
                with self.assertRaisesRegex(OSError, 'failed write'):
                    ppo.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_rejects_policy_checkpoint_as_ppo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.safetensors'
            self.make_ppo().actor_critic.save(path)
            with self.assertRaisesRegex(ValueError, 'unsupported PPO checkpoint format'):
                PPO.load(path)

    def test_policy_round_trip(self):
        observations = np.linspace(-0.5, 0.5, 32, dtype=np.float32).reshape(4, 8)
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                model = trainer.ActorCritic(8, 2, 16, 2, 1)
                if trainer is lander:
                    model.log_std.assign(lander.Tensor([-0.2, 0.4])).realize()
                    model.observation_scale.assign(model.observation_scale * 2).realize()
                    model.refresh_inference()
                else:
                    with torch.no_grad():
                        model.log_std.copy_(torch.tensor([-0.2, 0.4]))
                        model.observation_scale.mul_(2)
                for _ in range(3):
                    expected = model.act(observations, deterministic=True)
                with tempfile.TemporaryDirectory() as directory:
                    suffix = 'safetensors' if trainer is lander else 'pt'
                    path = Path(directory) / f'policy.{suffix}'
                    model.save(path)
                    restored = trainer.ActorCritic.load(path)
                    self.assertEqual(restored.config, model.config)
                    if trainer is lander:
                        self.assertIsNone(restored.actor_jit.captured)
                    for name, value in model.state_dict().items():
                        np.testing.assert_array_equal(restored.state_dict()[name].numpy(), value.numpy())
                    for actual, original in zip(restored.act(observations, deterministic=True), expected):
                        np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-6)
                    if trainer is lander:
                        np.testing.assert_array_equal(restored.action_std, model.action_std)
