import unittest

import numpy as np
import torch

import lander
import reference_lander


class ParityTest(unittest.TestCase):
    def test_ppo_loss_and_gradients(self):
        architecture = dict(state_dim=8, action_dim=2, hidden_dim=128, actor_layers=4, critic_layers=4)
        tiny_model = lander.ActorCritic(**architecture)
        torch_model = reference_lander.ActorCritic(**architecture).cpu()
        tiny_params = lander.nn.state.get_parameters(tiny_model.actor) + [tiny_model.log_std]
        tiny_params += lander.nn.state.get_parameters(tiny_model.critic)
        torch_params = list(torch_model.actor.parameters()) + [torch_model.log_std] + list(torch_model.critic.parameters())
        self.assertEqual(len(tiny_params), len(torch_params))
        random = np.random.default_rng(0)
        with torch.no_grad():
            for tiny, reference in zip(tiny_params, torch_params):
                values = (random.standard_normal(tuple(reference.shape)) * 0.05).astype(np.float32)
                tiny.assign(lander.Tensor(values)).realize()
                reference.copy_(torch.from_numpy(values))

        states = random.standard_normal((8, 8), dtype=np.float32)
        actions = random.uniform(-2, 2, (8, 2)).astype(np.float32)
        ratios = np.array([0.5, 0.9, 1.1, 1.5] * 2, dtype=np.float32)
        with torch.no_grad():
            means, stds, values = torch_model(torch.from_numpy(states))
            old_logprobs = torch.distributions.Normal(means, stds).log_prob(torch.from_numpy(actions)).sum(-1).numpy()
        tiny_means, tiny_log_std, tiny_values = tiny_model(lander.Tensor(states))
        np.testing.assert_allclose(tiny_means.numpy(), means.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(tiny_log_std.exp().numpy(), stds[0].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(tiny_values.numpy(), values.numpy(), rtol=1e-5, atol=1e-6)
        batch = [states, actions, old_logprobs - np.log(ratios),
                 np.array([-1] * 4 + [1] * 4, dtype=np.float32), random.standard_normal(8, dtype=np.float32)]
        settings = dict(pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                        eps_clip=0.2, batch_size=8, vf_coef=1.0, entropy_coef=0.001)
        tiny_ppo = lander.PPO(tiny_model, **settings)
        torch_ppo = reference_lander.PPO(torch_model, **settings)
        with lander.Context(TRAINING=1):
            actual = tiny_ppo.compute_loss(*[lander.Tensor(x) for x in batch])
            actual.backward()
        expected = torch_ppo.compute_loss(*[torch.from_numpy(x) for x in batch])
        expected.backward()
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
        for index, (tiny, reference) in enumerate(zip(tiny_params, torch_params)):
            with self.subTest(parameter=index):
                np.testing.assert_allclose(tiny.grad.numpy(), reference.grad.numpy(), rtol=1e-5, atol=1e-6)

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
