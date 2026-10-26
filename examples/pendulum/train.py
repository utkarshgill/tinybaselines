import argparse
from contextlib import ExitStack
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback

import numpy as np
from tinygrad.helpers import NUM_CPU_THREADS

from tinybaselines import GaussianActorCritic, PPO, evaluate, learn
from .env import make_env


def run(output, total_timesteps=200_000):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'metrics.jsonl').open('x') as metrics, ExitStack() as resources:
        def record(event, **fields):
            line = json.dumps(dict(event=event, seconds=perf_counter() - started, **fields), allow_nan=False)
            print(line, file=metrics, flush=True)
            print(line, flush=True)

        env = make_env(4)
        resources.callback(env.close)
        eval_env = make_env(4)
        resources.callback(eval_env.close)
        try:
            policy = GaussianActorCritic(3, 1, 64, 2, 2, observation_scale=[1, 1, 0.125])
            ppo = PPO(policy, pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=10,
                      eps_clip=0.2, batch_size=256, vf_coef=0.5, entropy_coef=0.0)
            try:
                commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True,
                                                 stderr=subprocess.DEVNULL).strip()
            except (FileNotFoundError, subprocess.CalledProcessError):
                commit = None
            record('config', environment='Pendulum-v1', source_commit=commit, num_envs=4,
                   requested_timesteps=total_timesteps, rollout_steps=2048, evaluation_episodes=100,
                   policy=policy.config, observation_scale=policy.observation_scale.numpy().tolist(), ppo=ppo.config,
                   device=str(ppo.device), dtype=str(policy.log_std.dtype), cpu_threads=NUM_CPU_THREADS.value,
                   beam=os.getenv('BEAM'), python=platform.python_version(), platform=platform.platform(),
                   versions={name: metadata.version(name) for name in ('tinygrad', 'numpy', 'gymnasium')},
                   backend_source=json.loads(metadata.distribution('tinygrad').read_text('direct_url.json') or '{}'))
            returns = evaluate(eval_env, policy, num_episodes=100)
            record('initial', mean_return=float(np.mean(returns)), episode_returns=returns)
            timesteps = 0
            for step in learn(env, ppo, total_timesteps=total_timesteps, rollout_steps=2048):
                timesteps = step.timesteps
                record('update', **step._asdict())
            checkpoint = output / 'ppo.safetensors'
            ppo.save(checkpoint)
            restored = PPO.load(checkpoint)
            returns = evaluate(eval_env, restored.actor_critic, num_episodes=100)
            record('final', timesteps=timesteps, mean_return=float(np.mean(returns)), episode_returns=returns,
                   checkpoint=checkpoint.name)
        except Exception as error:
            record('error', error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on Pendulum-v1 with four environments.')
    parser.add_argument('--output', type=Path, default=Path('runs/pendulum'))
    parser.add_argument('--timesteps', type=int, default=200_000)
    args = parser.parse_args()
    run(args.output, args.timesteps)
