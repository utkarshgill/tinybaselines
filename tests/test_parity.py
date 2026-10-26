import unittest

import numpy as np
import torch

import lander
import reference_lander


class ParityTest(unittest.TestCase):
    def test_adam_with_identical_gradients(self):
        tiny_parameter = lander.Tensor.zeros(3).is_param_()
        torch_parameter = torch.nn.Parameter(torch.zeros(3))
        tiny_optimizer = lander.nn.optim.Adam([tiny_parameter], lr=1e-3)
        torch_optimizer = torch.optim.Adam([torch_parameter], lr=1e-3)

        @lander.TinyJit
        @lander.Context(TRAINING=1)
        def step(gradient):
            tiny_parameter.grad = gradient
            return tiny_parameter.realize(*tiny_optimizer.schedule_step())

        gradient = lander.Tensor.zeros(3).realize()
        for scale in (1.0, -2.0, 0.5, 1.0):
            values = np.array([2.25e-9, 0.5, 0], dtype=np.float32) * scale
            gradient.assign(lander.Tensor(values)).realize()
            torch_parameter.grad = torch.from_numpy(values.copy())
            step(gradient)
            torch_optimizer.step()
            np.testing.assert_allclose(tiny_parameter.numpy(), torch_parameter.detach().numpy(), rtol=0, atol=1e-7)
        self.assertIsNotNone(step.captured)

    def test_gradient_clipping(self):
        for scale in (1.0, 0.01, 0.0):
            with self.subTest(scale=scale):
                tiny_params = [lander.Tensor.zeros(2).is_param_() for _ in range(3)]
                torch_params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
                for tiny, reference, values in zip(tiny_params, torch_params, ([3, 4], [0, 12])):
                    gradient = np.asarray(values, np.float32) * scale
                    tiny.grad = lander.Tensor(gradient)
                    reference.grad = torch.from_numpy(gradient.copy())
                lander.clip_grad_norm(tiny_params, 0.5)
                torch.nn.utils.clip_grad_norm_(torch_params, 0.5)
                for tiny, reference in zip(tiny_params[:2], torch_params[:2]):
                    np.testing.assert_allclose(tiny.grad.numpy(), reference.grad.numpy(), rtol=1e-5, atol=1e-6)
                self.assertIsNone(tiny_params[-1].grad)

    def test_ppo_training_steps(self):
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
        tiny_batch = [lander.Tensor(x).realize() for x in batch]
        torch_batch = [torch.from_numpy(x.copy()) for x in batch]
        with lander.Context(TRAINING=1):
            actual = tiny_ppo.compute_loss(*tiny_batch)
            actual.backward()
        expected = torch_ppo.compute_loss(*torch_batch)
        expected.backward()
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
        for index, (tiny, reference) in enumerate(zip(tiny_params, torch_params)):
            with self.subTest(parameter=index):
                np.testing.assert_allclose(tiny.grad.numpy(), reference.grad.numpy(), rtol=1e-5, atol=1e-6)
        for step in range(4):
            returns = batch[-1] + np.float32(step * 0.1)
            tiny_batch[-1].assign(lander.Tensor(returns)).realize()
            torch_batch[-1].copy_(torch.from_numpy(returns))
            actual = tiny_ppo.train_step(*tiny_batch)
            expected = torch_ppo.train_step(*torch_batch)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
            for index, (tiny, reference) in enumerate(zip(tiny_params, torch_params)):
                with self.subTest(step=step, parameter=index):
                    # Near-zero gradient roundoff is amplified by Adam.
                    np.testing.assert_allclose(tiny.numpy(), reference.detach().numpy(), rtol=0, atol=2e-6)

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
