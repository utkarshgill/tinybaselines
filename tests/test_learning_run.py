import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import reference_lander
import run_learning


class LearningRunTest(unittest.TestCase):
    def test_records_steps_and_evaluates_saved_policy(self):
        policy = SimpleNamespace(log_std=torch.zeros(2), config={}, save=Mock())
        ctx = SimpleNamespace(ac=policy, eval_env=object(), cleanup=Mock())
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
            self.assertEqual([entry['steps'] for entry in events[1:]], [20_000, 40_000, 40_000])
            self.assertEqual([entry['score'] for entry in events[1:]], [100.0, 205.0, 210.0])
            self.assertTrue(events[-1]['stopped_on_threshold'])
