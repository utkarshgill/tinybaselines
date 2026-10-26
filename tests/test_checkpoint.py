import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import lander
import reference_lander


class CheckpointTest(unittest.TestCase):
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
