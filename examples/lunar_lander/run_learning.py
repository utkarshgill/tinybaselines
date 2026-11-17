import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback


def run(framework, output, *, seed=None, budget=None):
    started = time.perf_counter()
    if seed is not None and not 0 <= seed < 2**32:
        raise ValueError('seed must be between 0 and 2**32 - 1')
    os.environ.update(NUM_ENVS='4', METAL='0', PLOT='0', RENDER='0', CUDA_VISIBLE_DEVICES='')
    trainer = importlib.import_module('examples.lunar_lander.train' if framework == 'tinygrad' else 'examples.lunar_lander.reference')
    if budget is not None and (budget <= 0 or budget % trainer.steps_per_epoch):
        raise ValueError(f'budget must be a positive multiple of {trainer.steps_per_epoch} transitions')
    epochs = trainer.max_epochs if budget is None else budget // trainer.steps_per_epoch
    if framework == 'tinygrad':
        from tinygrad.helpers import NUM_CPU_THREADS
    settings = {name: getattr(trainer, name) for name in (
        'num_envs', 'steps_per_epoch', 'max_epochs', 'batch_size', 'K_epochs', 'pi_lr', 'vf_lr',
        'gamma', 'gae_lambda', 'eps_clip', 'vf_coef', 'entropy_coef', 'eval_interval', 'eval_episodes', 'solved_threshold',
    )}
    settings['max_epochs'] = epochs
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'metrics.jsonl').open('x') as metrics:
        def record(**entry):
            line = json.dumps(dict(seconds=time.perf_counter() - started, **entry), allow_nan=False)
            print(line, file=metrics, flush=True)
            print(line, flush=True)

        ctx = trainer.TrainingContext(seed=seed, epochs=epochs)
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True
            ).strip()
            record(event='config', framework=framework, environment=trainer.ENV_NAME,
                   source_commit=commit,
                   device=str(ctx.ac.log_std.device), dtype=str(ctx.ac.log_std.dtype),
                   platform=platform.platform(), python=platform.python_version(),
                   versions={name: metadata.version(name) for name in ('gymnasium', 'numpy', framework)},
                   backend_source=json.loads(metadata.distribution(framework).read_text('direct_url.json') or '{}'),
                   settings=settings, seed=seed, budget=budget, architecture=ctx.ac.config, beam=os.getenv('BEAM'),
                   torch_threads=trainer.torch.get_num_threads() if framework == 'torch' else None,
                   tinygrad_cpu_threads=NUM_CPU_THREADS.value if framework == 'tinygrad' else None)
            for epoch in range(epochs):
                epoch_started = time.perf_counter()
                solved = trainer.train_one_epoch(epoch, ctx)
                epoch_seconds = time.perf_counter() - epoch_started
                rollout_seconds, update_seconds = ctx.rollout_times[-1], ctx.update_times[-1]
                diagnostics = dict(approx_kl=ctx.last_step.approx_kl,
                                   clip_fraction=ctx.last_step.clip_fraction) if framework == 'tinygrad' else {}
                record(event='epoch', steps=(epoch + 1) * trainer.steps_per_epoch, seconds_total=epoch_seconds,
                       rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                       other_seconds=epoch_seconds - rollout_seconds - update_seconds, **diagnostics)
                if epoch % trainer.eval_interval == 0:
                    record(event='evaluation', steps=(epoch + 1) * trainer.steps_per_epoch, score=ctx.last_eval_stochastic)
                if solved and budget is None:
                    break
            seconds_to_stop = time.perf_counter() - started
            policy_path = output / ('policy.safetensors' if framework == 'tinygrad' else 'policy.pt')
            ctx.ac.save(policy_path)
            checkpoint_path = output / 'ppo.safetensors' if framework == 'tinygrad' else None
            if checkpoint_path is not None:
                ctx.ppo.save(checkpoint_path)
                restored = trainer.PPO.load(checkpoint_path).actor_critic
            else:
                restored = trainer.ActorCritic.load(policy_path)
            score = trainer.evaluate_policy(restored, env=ctx.eval_env, deterministic=False)
            record(event='final', steps=(epoch + 1) * trainer.steps_per_epoch, score=score,
                   seconds_to_stop=seconds_to_stop, stopped_on_threshold=solved and budget is None, policy=policy_path.name,
                   checkpoint=checkpoint_path.name if checkpoint_path is not None else None)
        except Exception as error:
            record(event='error', error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            ctx.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the four-environment CPU LunarLander experiment.')
    parser.add_argument('framework', choices=('tinygrad', 'torch'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, help='Seed model, sampling, and environments.')
    parser.add_argument('--budget', type=int, help='Train for exactly this many transitions in full rollouts, even after solving.')
    args = parser.parse_args()
    run(args.framework, args.output, seed=args.seed, budget=args.budget)
