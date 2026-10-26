import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.lunar_lander import train as lander, reference as reference_lander
from tinybaselines import GaussianActorCritic, PPO


class CheckpointTest(unittest.TestCase):
    def test_ppo_state_contains_policy_and_adam_buffers(self):
        policy = GaussianActorCritic(3, 1, 4, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=2e-3, gamma=0.97, lamda=0.89, K_epochs=2,
                  eps_clip=0.17, batch_size=4, vf_coef=0.8, entropy_coef=0.03)
        ppo.gamma = 0.9
        ppo.pi_optimizer.lr.assign(lander.Tensor([0.005])).realize()
        self.assertEqual(ppo.config['gamma'], 0.9)
        self.assertAlmostEqual(ppo.config['pi_lr'], 0.005)
        state = ppo.state_dict()
        for name, tensor in policy.state_dict().items():
            self.assertIs(state['policy.' + name], tensor)
        for name in ('pi_optimizer', 'vf_optimizer'):
            optimizer = getattr(ppo, name)
            for key in ('lr', 'b1_t', 'b2_t'):
                self.assertIs(state[name + '.' + key], getattr(optimizer, key))
            for key in ('m', 'v'):
                for index, tensor in enumerate(getattr(optimizer, key)):
                    self.assertIs(state[f'{name}.{key}.{index}'], tensor)
        self.assertFalse(any('.params.' in name for name in state))

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
