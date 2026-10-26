import unittest

import numpy as np
import torch

import lander


class ParityTest(unittest.TestCase):
    def test_gaussian_log_prob_and_gradients(self):
        means = np.array([[0.2, -0.3], [1.5, -1.5]], dtype=np.float32)
        log_std = np.array([-0.7, 0.2], dtype=np.float32)
        actions = np.array([[0.4, 0.8], [2, -2]], dtype=np.float32)
        tiny_mean = lander.Tensor(means).is_param_()
        tiny_log_std = lander.Tensor(log_std).is_param_()
        torch_mean = torch.tensor(means, requires_grad=True)
        torch_log_std = torch.tensor(log_std, requires_grad=True)
        with lander.Context(TRAINING=1):
            actual = lander.normal_log_prob(tiny_mean, tiny_log_std, lander.Tensor(actions))
            actual.sum().backward()
        expected = torch.distributions.Normal(torch_mean, torch_log_std.exp()).log_prob(torch.tensor(actions)).sum(-1)
        expected.sum().backward()
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
        for tiny, reference in ((tiny_mean, torch_mean), (tiny_log_std, torch_log_std)):
            np.testing.assert_allclose(tiny.grad.numpy(), reference.grad.numpy(), rtol=1e-5, atol=1e-6)
