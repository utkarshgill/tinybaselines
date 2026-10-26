import gymnasium as gym
import numpy as np
import os
import warnings

METAL = bool(int(os.getenv('METAL', '0')))
os.environ['DEV'] = 'METAL' if METAL else 'CPU'
if not METAL:
    os.environ.setdefault('BEAM', '1')

from tinygrad import Tensor, TinyJit, nn, Context
from tqdm import trange
from .config import ENV_NAME, OBS_SCALE
from tinybaselines import GaussianActorCritic, PPO, evaluate, learn
from tinybaselines.ppo import clip_grad_norm, normal_log_prob
from tinybaselines.rollout import RolloutBatch, RolloutState, rollout, track_episode_returns

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

state_dim, action_dim = 8, 2
num_envs = int(os.getenv('NUM_ENVS', 4))

max_epochs, steps_per_epoch = 100, 20_000
log_interval, eval_interval = 5, 5
eval_episodes = 100
batch_size, K_epochs = 10_000, 20
hidden_dim = 128
actor_layers, critic_layers = 4, 4
pi_lr, vf_lr = 1e-3, 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
vf_coef, entropy_coef = 1.0, 0.001
solved_threshold = 200

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
RENDER_EPISODES = int(os.getenv('RENDER_EPISODES', '3'))

device = 'METAL' if METAL else 'CPU'

if PLOT:
    import matplotlib.pyplot as plt


def make_env(n, render=False, asynchronous=True):
    render_mode = 'human' if render else None
    env_class = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return env_class(
        [lambda: gym.make(ENV_NAME, render_mode=render_mode) for _ in range(n)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def update_plot(ax, returns, threshold):
    ax.clear()
    ax.plot(returns, alpha=0.3, label='Episode Returns')
    if len(returns) >= 100:
        ma = np.convolve(returns, np.ones(100) / 100, mode='valid')
        ax.plot(range(99, len(returns)), ma, label='100-ep MA', linewidth=2)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({threshold})')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    plt.pause(0.01)


class ActorCritic(GaussianActorCritic):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super().__init__(state_dim, action_dim, hidden_dim, actor_layers, critic_layers, observation_scale=OBS_SCALE)


class TrainingContext:
    def __init__(self):
        self.ac = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers)
        self.ppo = PPO(self.ac, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef)

        self.env = make_env(num_envs)
        self.training = learn(self.env, self.ppo, total_timesteps=max_epochs * steps_per_epoch, rollout_steps=steps_per_epoch)
        self.eval_env = make_env(min(num_envs, eval_episodes), asynchronous=False)
        self.all_episode_returns = []
        self.last_eval_stochastic = float('-inf')
        self.pbar = trange(max_epochs, desc='Training', unit='epoch')
        self.rollout_times = []
        self.update_times = []

        if PLOT:
            plt.ion()
            _, self.ax = plt.subplots()
        else:
            self.ax = None

    def cleanup(self):
        self.env.close()
        self.eval_env.close()
        self.pbar.close()
        if PLOT:
            plt.ioff()
            plt.show()


def train_one_epoch(epoch, ctx):
    step = next(ctx.training)
    ep_rets = step.episode_returns
    ctx.rollout_times.append(step.rollout_seconds)
    ctx.update_times.append(step.update_seconds)

    ctx.all_episode_returns.extend(ep_rets)
    ctx.pbar.update(1)

    if epoch % eval_interval == 0:
        ctx.last_eval_stochastic = evaluate_policy(ctx.ac, env=ctx.eval_env, deterministic=False)
        if RENDER:
            evaluate_policy(ctx.ac, render=True, num_episodes=RENDER_EPISODES, deterministic=False)

    if epoch % log_interval == 0:
        s = np.exp(ctx.ac.log_std.clip(-5, 2).numpy())
        rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
        update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
        total_ms = rollout_ms + update_ms
        ctx.pbar.write(f'Epoch {epoch:3d}  n_ep={len(ep_rets):3d}  ret={np.mean(ep_rets):7.1f}±{np.std(ep_rets):5.1f}  eval_stoch={ctx.last_eval_stochastic:6.1f}  σ=[{s[0]:.2f} {s[1]:.2f}]  ⏱ {total_ms:.0f}ms (rollout:{rollout_ms:.0f}ms update:{update_ms:.0f}ms)')

    if PLOT and epoch % (log_interval * 2) == 0:
        update_plot(ctx.ax, ctx.all_episode_returns, solved_threshold)

    if ctx.last_eval_stochastic >= solved_threshold:
        ctx.pbar.write(f'\n{"=" * 60}\nSOLVED at epoch {epoch}! eval_stoch={ctx.last_eval_stochastic:.1f} ≥ {solved_threshold}\n{"=" * 60}')
        if RENDER:
            evaluate_policy(ctx.ac, render=True, num_episodes=RENDER_EPISODES, deterministic=False)
        return True

    return False


def train():
    ctx = TrainingContext()
    try:
        for epoch in range(max_epochs):
            if train_one_epoch(epoch, ctx):
                break
    finally:
        ctx.cleanup()


def evaluate_policy(actor_critic, num_episodes=eval_episodes, render=False, env=None, deterministic=True):
    close_env = env is None
    if env is None:
        env = make_env(1 if render else num_episodes, render)
    try:
        return float(np.mean(evaluate(env, actor_critic, num_episodes, deterministic=deterministic)))
    finally:
        if close_env:
            env.close()


if __name__ == '__main__':
    print(f'Using Tinygrad {device} device')
    train()
