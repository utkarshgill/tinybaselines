import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import torch

from examples.lunar_lander import reference as reference_lander, train, run_learning
from examples.lunar_lander.config import SeededReset
from tinybaselines import TrainingStep


class LearningRunTest(unittest.TestCase):
    def test_seed_applies_to_first_reset_only(self):
        def reset_seeds(seed, stream):
            env = Mock(spec=gym.vector.VectorEnv)
            wrapped = SeededReset(env, seed, stream)
            wrapped.reset()
            wrapped.reset()
            return [call.kwargs['seed'] for call in env.reset.call_args_list]

        seeds = reset_seeds(0, 0)
        self.assertEqual(seeds, reset_seeds(0, 0))
        self.assertIsInstance(seeds[0], int)
        self.assertIsNone(seeds[1])
        self.assertNotEqual(seeds[0], reset_seeds(0, 1)[0])
        self.assertNotEqual(seeds[0], reset_seeds(1, 0)[0])

    def test_fixed_budget_continues_after_threshold(self):
        policy = SimpleNamespace(log_std=train.Tensor.zeros(2), config={}, save=Mock())
        ctx = SimpleNamespace(ac=policy, ppo=SimpleNamespace(save=Mock()), eval_env=object(), cleanup=Mock(),
                              rollout_times=[0.0], update_times=[0.0], last_eval_stochastic=205.0,
                              last_step=TrainingStep(20_000, [], 0.0, 0.0, 0.03, 0.2))
        context, epoch = Mock(return_value=ctx), Mock(return_value=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.dict(os.environ), patch.multiple(train, max_epochs=1, TrainingContext=context,
                    train_one_epoch=epoch, evaluate_policy=Mock(return_value=210.0)), patch.object(train.PPO, 'load'):
                run_learning.run('tinygrad', output, seed=0, budget=40_000)
            events = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        context.assert_called_once_with(seed=0, epochs=2)
        self.assertEqual([call.args[0] for call in epoch.call_args_list], [0, 1])
        self.assertEqual((events[0]['seed'], events[0]['budget'], events[0]['settings']['max_epochs']), (0, 40_000, 2))
        self.assertEqual(events[-1]['steps'], 40_000)
        self.assertFalse(events[-1]['stopped_on_threshold'])

    def test_rejects_invalid_seed_and_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            output = Path(directory) / 'run'
            with self.assertRaisesRegex(ValueError, 'seed'):
                run_learning.run('tinygrad', output, seed=-1)
            for budget in (0, 20_001):
                with self.assertRaisesRegex(ValueError, 'positive multiple'):
                    run_learning.run('tinygrad', output, budget=budget)
            self.assertFalse(output.exists())

    def test_saves_and_evaluates_ppo_checkpoint(self):
        policy = SimpleNamespace(log_std=train.Tensor.zeros(2), config={}, save=Mock())
        ctx = SimpleNamespace(ac=policy, ppo=SimpleNamespace(save=Mock()), eval_env=object(), cleanup=Mock(),
                              rollout_times=[], update_times=[], all_episode_returns=[], pbar=Mock(),
                              training=iter([TrainingStep(20_000, [1.0], 0.1, 0.2, 0.03, 0.2)]))
        restored = object()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.dict(os.environ), patch.multiple(train, max_epochs=1, PLOT=False, RENDER=False,
                    TrainingContext=Mock(return_value=ctx), evaluate_policy=Mock(side_effect=[205.0, 210.0])):
                with patch.object(train.PPO, 'load', return_value=SimpleNamespace(actor_critic=restored)) as load:
                    run_learning.run('tinygrad', output)
                    policy.save.assert_called_once_with(output / 'policy.safetensors')
                    ctx.ppo.save.assert_called_once_with(output / 'ppo.safetensors')
                    load.assert_called_once_with(output / 'ppo.safetensors')
                    train.evaluate_policy.assert_called_with(restored, env=ctx.eval_env, deterministic=False)
            events = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            epoch, = [entry for entry in events if entry['event'] == 'epoch']
            self.assertEqual((epoch['approx_kl'], epoch['clip_fraction']), (0.03, 0.2))
            self.assertEqual(events[-1]['checkpoint'], 'ppo.safetensors')
            self.assertEqual(events[-1]['policy'], 'policy.safetensors')
            ctx.cleanup.assert_called_once()

    def test_records_failure_and_closes_environments(self):
        policy = SimpleNamespace(log_std=torch.zeros(2), config={})
        ctx = SimpleNamespace(ac=policy, cleanup=Mock())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.dict(os.environ), patch.multiple(reference_lander,
                    TrainingContext=Mock(return_value=ctx), train_one_epoch=Mock(side_effect=RuntimeError('failed update'))):
                with self.assertRaisesRegex(RuntimeError, 'failed update'):
                    run_learning.run('torch', output)
            failure = json.loads((output / 'metrics.jsonl').read_text().splitlines()[-1])
            self.assertEqual(failure['event'], 'error')
            self.assertEqual(failure['error_type'], 'RuntimeError')
            self.assertIn('failed update', failure['traceback'])
            ctx.cleanup.assert_called_once()

    def test_records_steps_and_evaluates_saved_policy(self):
        policy = SimpleNamespace(log_std=torch.zeros(2), config={}, save=Mock())
        ctx = SimpleNamespace(ac=policy, eval_env=object(), cleanup=Mock(), rollout_times=[0.0], update_times=[0.0])
        restored = object()

        def train(epoch, ctx):
            ctx.last_eval_stochastic = (100.0, 205.0)[epoch]
            return epoch == 1

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch.dict(os.environ), patch.multiple(reference_lander, max_epochs=3, eval_interval=1,
                    TrainingContext=Mock(return_value=ctx), train_one_epoch=train,
                    evaluate_policy=Mock(return_value=210.0)):
                with patch.object(reference_lander.ActorCritic, 'load', return_value=restored) as load:
                    run_learning.run('torch', output)
                    policy.save.assert_called_once_with(output / 'policy.pt')
                    load.assert_called_once_with(output / 'policy.pt')
                    reference_lander.evaluate_policy.assert_called_once_with(restored, env=ctx.eval_env, deterministic=False)
            ctx.cleanup.assert_called_once()
            events = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            scores = [entry for entry in events if entry['event'] in ('evaluation', 'final')]
            self.assertEqual([entry['steps'] for entry in scores], [20_000, 40_000, 40_000])
            self.assertEqual([entry['score'] for entry in scores], [100.0, 205.0, 210.0])
            epochs = [entry for entry in events if entry['event'] == 'epoch']
            self.assertEqual([entry['steps'] for entry in epochs], [20_000, 40_000])
            for entry in epochs:
                self.assertTrue({'approx_kl', 'clip_fraction'}.isdisjoint(entry))
                self.assertAlmostEqual(entry['seconds_total'],
                                       entry['rollout_seconds'] + entry['update_seconds'] + entry['other_seconds'])
                self.assertGreaterEqual(entry['other_seconds'], 0)
            self.assertTrue(events[-1]['stopped_on_threshold'])
