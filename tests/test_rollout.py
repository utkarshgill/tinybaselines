import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import lander
import reference_lander


class RolloutTest(unittest.TestCase):
    def test_episode_crosses_rollout_boundary(self):
        actions = np.zeros((4, 2), dtype=np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                observations = np.zeros((4, 8), dtype=np.float32)

                def step(actions):
                    observations[:] += 1
                    ended = observations[:, 0] == 3
                    final_obs = observations.copy()
                    observations[ended] = 0
                    return observations, np.ones(4), ended, np.zeros(4, bool), {'final_obs': final_obs}

                env = SimpleNamespace(num_envs=4, reset=Mock(return_value=(observations, {})), step=step)
                state = trainer.RolloutState(env.reset()[0], np.zeros(4))
                first = trainer.rollout(env, policy, num_steps=8, state=state)
                self.assertEqual(first[-1], [])
                np.testing.assert_array_equal(state.episode_returns, np.full(4, 2))
                ctx = SimpleNamespace(env=env, ac=policy, rollout_state=state, ppo=Mock(), pbar=Mock(),
                                      rollout_times=[], update_times=[], all_episode_returns=[], last_eval_stochastic=-np.inf)
                with patch.object(trainer, 'steps_per_epoch', 8):
                    trainer.train_one_epoch(1, ctx)
                env.reset.assert_called_once()
                np.testing.assert_array_equal(first[0][:, 0, 0], [0, 1])
                np.testing.assert_array_equal(ctx.ppo.update.call_args.args[0][:, 0, 0], [2, 0])
                np.testing.assert_array_equal(ctx.all_episode_returns, np.full(4, 3))
                np.testing.assert_array_equal(state.observations, np.ones((4, 8)))
                np.testing.assert_array_equal(state.episode_returns, np.ones(4))

    def test_episode_endings_remain_distinct(self):
        terminated = np.array([True, False, True, False])
        truncated = np.array([False, True, True, False])
        states = np.zeros((4, 8), dtype=np.float32)
        final_obs = np.full((4, 8), 7, dtype=np.float32)
        actions = np.zeros((4, 2), dtype=np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        env = SimpleNamespace(num_envs=4, reset=lambda: (states, {}),
                              step=lambda actions: (states, np.ones(4), terminated, truncated, {'final_obs': final_obs}))
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                result = trainer.rollout(env, policy, num_steps=4)
                np.testing.assert_array_equal(result[3][0], terminated)
                np.testing.assert_array_equal(result[4][0], truncated)
                np.testing.assert_array_equal(result[5][0, :3], final_obs[:3])
                np.testing.assert_array_equal(result[5][0, 3], states[3])
                self.assertEqual(len(result[-1]), 3)

    def test_each_step_is_a_transition(self):
        make = lander.gym.make
        actions = np.zeros((4, 2), dtype=np.float32)
        policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                with patch.object(trainer.gym, 'make', side_effect=lambda *a, **kw: make(*a, max_episode_steps=1, **kw)):
                    env = trainer.make_env(4, asynchronous=False)
                try:
                    _, _, _, terminated, truncated, _, episodes = trainer.rollout(env, policy, num_steps=8)
                    np.testing.assert_array_equal(terminated, np.zeros((2, 4), bool))
                    np.testing.assert_array_equal(truncated, np.ones((2, 4), bool))
                    self.assertEqual(len(episodes), 8)
                finally:
                    env.close()

    def test_next_observations_are_snapshots(self):
        for trainer in (lander, reference_lander):
            with self.subTest(trainer=trainer.__name__):
                states = np.zeros((4, 8), dtype=np.float32)

                def step(actions):
                    states[:] += 1
                    return states, np.ones(4), np.zeros(4, bool), np.zeros(4, bool), {}

                env = SimpleNamespace(num_envs=4, reset=lambda: (states, {}), step=step)
                actions = np.zeros((4, 2), dtype=np.float32)
                policy = SimpleNamespace(act=lambda obs, deterministic: (actions, actions))
                obs, _, _, _, _, next_obs, _ = trainer.rollout(env, policy, num_steps=8)
                np.testing.assert_array_equal(obs[:, 0, 0], [0, 1])
                np.testing.assert_array_equal(next_obs[:, 0, 0], [1, 2])
                states[:] = 99
                np.testing.assert_array_equal(next_obs[:, 0, 0], [1, 2])

    def test_bootstrap_values(self):
        cases = {
            'termination': ([[0, 1]], [[2, 3]], [[0, 1]], [[0, 0]], [[10, 20]], [9.9, 1]),
            'rollout_cutoff': ([[0], [0]], [[2], [3]], [[0], [0]], [[0], [0]], [[3], [10]], [9.45945, 9.9]),
            'time_limit': ([[1], [100]], [[2], [3]], [[0], [1]], [[1], [0]], [[10], [20]], [10.9, 100]),
            'both_flags': ([[1, 1]], [[2, 2]], [[1, 0]], [[1, 1]], [[10, 10]], [1, 10.9]),
        }
        for trainer in (lander, reference_lander):
            for name, case in cases.items():
                rewards, values, terminated, truncated, next_values, expected = case
                with self.subTest(trainer=trainer.__name__, case=name):
                    ppo = trainer.PPO.__new__(trainer.PPO)
                    ppo.gamma, ppo.lamda = 0.99, 0.95
                    arrays = [
                        np.asarray(rewards, np.float32), np.asarray(values, np.float32),
                        np.asarray(terminated, bool), np.asarray(truncated, bool),
                        np.asarray(next_values, np.float32),
                    ]
                    if trainer is reference_lander:
                        arrays = [trainer.torch.from_numpy(x) for x in arrays]
                    advantages, returns = ppo.compute_advantages(*arrays)
                    np.testing.assert_allclose(returns, expected, rtol=1e-6, atol=1e-6)
                    expected_advantages = np.asarray(expected) - np.asarray(values).reshape(-1)
                    expected_advantages = (expected_advantages - expected_advantages.mean()) / (expected_advantages.std() + 1e-8)
                    np.testing.assert_allclose(advantages, expected_advantages, rtol=1e-5, atol=1e-5)
