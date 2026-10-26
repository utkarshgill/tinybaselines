import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from examples.lunar_lander import reference as reference_lander, run_learning


class LearningRunTest(unittest.TestCase):
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
                self.assertAlmostEqual(entry['seconds_total'],
                                       entry['rollout_seconds'] + entry['update_seconds'] + entry['other_seconds'])
                self.assertGreaterEqual(entry['other_seconds'], 0)
            self.assertTrue(events[-1]['stopped_on_threshold'])
