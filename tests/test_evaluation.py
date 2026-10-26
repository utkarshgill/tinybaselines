from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tinybaselines import evaluate


class EvaluationTest(unittest.TestCase):
    def test_example_reports_mean_and_respects_environment_ownership(self):
        from examples.lunar_lander import train

        for owns_env in (False, True):
            with self.subTest(owns_env=owns_env):
                env, policy = Mock(), Mock()
                with patch.object(train, 'make_env', return_value=env), \
                     patch.object(train, 'evaluate', return_value=[1.0, 3.0]) as evaluate:
                    score = train.evaluate_policy(policy, num_episodes=2, env=None if owns_env else env)
                self.assertEqual(score, 2.0)
                evaluate.assert_called_once_with(env, policy, 2, deterministic=True)
                self.assertEqual(env.close.call_count, int(owns_env))

    def test_example_closes_owned_environment_on_failure(self):
        from examples.lunar_lander import train

        for owns_env in (False, True):
            with self.subTest(owns_env=owns_env):
                env = SimpleNamespace(num_envs=1, reset=Mock(return_value=(np.zeros((1, 3)), {})), close=Mock())
                policy = SimpleNamespace(act=Mock(side_effect=RuntimeError('failed inference')))
                with patch.object(train, 'make_env', return_value=env):
                    with self.assertRaisesRegex(RuntimeError, 'failed inference'):
                        train.evaluate_policy(policy, num_episodes=1, env=None if owns_env else env)
                self.assertEqual(env.close.call_count, int(owns_env))

    def test_episode_quotas_and_action_modes(self):
        for deterministic in (False, True):
            for episodes, expected in ((2, [300, 1000]), (7, [1, 10, 10, 300, 300, 1000, 1000]),
                                       (100, np.repeat([1, 10, 300, 1000], 25))):
                with self.subTest(deterministic=deterministic, episodes=episodes):
                    ages = np.zeros(4, dtype=int)
                    observations = np.zeros((4, 3), np.float32)
                    actions = np.zeros((4, 1), np.float32)
                    policy = SimpleNamespace(act=Mock(return_value=(actions, actions)))

                    def step(actions):
                        ages[:] += 1
                        ended = ages == [1, 2, 3, 4]
                        ages[ended] = 0
                        return observations, np.array([1, 5, 100, 250]), ended, np.zeros(4, bool), {}

                    env = SimpleNamespace(num_envs=4, reset=Mock(return_value=(observations, {})),
                                          step=step, close=Mock())
                    returns = evaluate(env, policy, episodes, deterministic=deterministic)
                    np.testing.assert_array_equal(sorted(returns), expected)
                    env.reset.assert_called_once()
                    env.close.assert_not_called()
                    for call in policy.act.call_args_list:
                        self.assertEqual(call.kwargs['deterministic'], deterministic)

    def test_rejects_nonpositive_episode_count(self):
        env, policy = Mock(), Mock()
        for episodes in (0, -1):
            with self.subTest(episodes=episodes):
                with self.assertRaisesRegex(ValueError, 'num_episodes must be positive'):
                    evaluate(env, policy, episodes)
        env.reset.assert_not_called()
        policy.act.assert_not_called()

    def test_stochastic_default_and_error_propagation(self):
        policy = SimpleNamespace(act=Mock(side_effect=RuntimeError('failed inference')))
        env = SimpleNamespace(num_envs=1, reset=Mock(return_value=(np.zeros((1, 3)), {})), close=Mock())
        with self.assertRaisesRegex(RuntimeError, 'failed inference'):
            evaluate(env, policy, 1)
        self.assertFalse(policy.act.call_args.kwargs['deterministic'])
        env.close.assert_not_called()
