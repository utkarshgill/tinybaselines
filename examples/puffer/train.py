import argparse
from contextlib import closing
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from tinybaselines import GaussianActorCritic, PPO, evaluate, learn
from .env import SquaredVectorEnv


def run(library, output, total_timesteps):
    output.mkdir(parents=True, exist_ok=False)
    policy = GaussianActorCritic(121, 2, 64, 2, 2)
    ppo = PPO(policy, pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=4,
              eps_clip=0.2, batch_size=256, vf_coef=0.5, entropy_coef=0.001)

    with (output / 'metrics.jsonl').open('x') as log:
        def record(event, **fields):
            line = json.dumps(dict(event=event, **fields), allow_nan=False)
            print(line, file=log, flush=True)
            print(line, flush=True)

        def score(event, actor):
            with closing(SquaredVectorEnv(library, num_envs=4)) as env:
                returns = list(map(float, evaluate(env, actor, num_episodes=100, deterministic=False)))
            assert len(returns) == 100 and all(value in (-1, 1) for value in returns)
            record(event, goals=returns.count(1), episodes=len(returns), episode_returns=returns)

        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        record('config', source_commit=commit or None, platform=platform.platform(), python=platform.python_version(),
               versions={name: version(name) for name in ('tinygrad', 'numpy')},
               library_sha256=sha256(library.read_bytes()).hexdigest(), device=ppo.device,
               num_envs=4, requested_timesteps=total_timesteps, rollout_steps=2048,
               deterministic=False, policy=policy.config, ppo=ppo.config)
        score('initial', policy)
        timesteps = 0
        with closing(SquaredVectorEnv(library, num_envs=4)) as env:
            started = perf_counter()
            for step in learn(env, ppo, total_timesteps=total_timesteps, rollout_steps=2048):
                timesteps = step.timesteps
                record('update', **step._asdict())
            training_seconds = perf_counter() - started
        checkpoint = output / 'ppo.safetensors'
        ppo.save(checkpoint)
        record('checkpoint', timesteps=timesteps, training_seconds=training_seconds, path=checkpoint.name)
        score('final', PPO.load(checkpoint).actor_critic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on headless Squared Continuous.')
    parser.add_argument('library', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timesteps', type=int, default=204_800)
    args = parser.parse_args()
    run(args.library, args.output, args.timesteps)
