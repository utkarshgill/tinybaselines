import gymnasium as gym
import numpy as np
import time
import os
import warnings

METAL = bool(int(os.getenv('METAL', '0')))
os.environ['DEV'] = 'METAL' if METAL else 'CPU'
if not METAL:
    os.environ.setdefault('BEAM', '1')

from tinygrad import Tensor, TinyJit, nn, Context
from tqdm import trange

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

env_name = 'LunarLanderContinuous-v3'
state_dim, action_dim = 8, 2
num_envs = int(os.getenv('NUM_ENVS', 100))

OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)

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
LOG_2PI = float(np.log(2.0 * np.pi))
NORMAL_ENTROPY_CONSTANT = float(0.5 * np.log(2.0 * np.pi * np.e))

if PLOT:
    import matplotlib.pyplot as plt


def make_env(n, render=False, asynchronous=True):
    render_mode = 'human' if render else None
    env_class = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return env_class(
        [lambda: gym.make(env_name, render_mode=render_mode) for _ in range(n)],
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


def track_episode_returns(done_mask, ep_returns, ep_rets):
    for idx in np.where(done_mask)[0]:
        ep_returns.append(ep_rets[idx])
        ep_rets[idx] = 0.0


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dim, layer_count):
        self.layers = [nn.Linear(input_dim, hidden_dim)]
        self.layers.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_count - 1))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)


class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        self.actor = MLP(state_dim, action_dim, hidden_dim, actor_layers)
        self.critic = MLP(state_dim, 1, hidden_dim, critic_layers)
        self.actor.layers[-1].weight.assign(self.actor.layers[-1].weight * 0.01).realize()
        self.actor.layers[-1].bias.assign(Tensor.zeros_like(self.actor.layers[-1].bias)).realize()
        self.log_std = Tensor.full((action_dim,), -0.7, device=device).realize()
        self.action_std = np.exp(self.log_std.numpy())
        self.jit_shape = None
        self.actor_jit = TinyJit(lambda state: self.actor(state).realize())

    def __call__(self, state):
        action_mean = self.actor(state)
        log_std = self.log_std.clip(-5, 2)
        value = self.critic(state)
        return action_mean, log_std, value

    def act(self, state, deterministic=False):
        state_tensor = Tensor(state.astype(np.float32) * OBS_SCALE, device=device)
        if self.jit_shape != state_tensor.shape:
            self.actor_jit.reset()
            self.jit_shape = state_tensor.shape
        mean = self.actor_jit(state_tensor).numpy()
        if deterministic:
            action = np.clip(mean, -1, 1)
            return action, mean
        sample = np.random.normal(mean, self.action_std).astype(np.float32)
        action = np.clip(sample, -1, 1)
        return action, sample

    def refresh_action_std(self):
        self.action_std = np.exp(self.log_std.clip(-5, 2).numpy())


def normal_log_prob(action_mean, log_std, actions):
    normalized = (actions - action_mean) / log_std.exp()
    return (-0.5 * (normalized.square() + 2.0 * log_std + LOG_2PI)).sum(axis=-1)


def clip_grad_norm(parameters, max_norm):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_norm = sum((gradient.square().sum() for gradient in gradients)).sqrt()
    scale = (max_norm / (total_norm + 1e-6)).clip(max_=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = parameter.grad * scale


class PPO:
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.actor_parameters = nn.state.get_parameters(actor_critic.actor) + [actor_critic.log_std]
        self.critic_parameters = nn.state.get_parameters(actor_critic.critic)
        self.pi_optimizer = nn.optim.Adam(self.actor_parameters, lr=pi_lr)
        self.vf_optimizer = nn.optim.Adam(self.critic_parameters, lr=vf_lr)
        self.optimizer = nn.optim.OptimizerGroup(self.pi_optimizer, self.vf_optimizer)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size
        self.vf_coef, self.entropy_coef = vf_coef, entropy_coef

    def compute_advantages(self, rewards, state_values, is_terminals):
        T, N = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        state_values_pad = np.concatenate([state_values, state_values[-1:]], axis=0)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)

    def compute_loss(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, log_std, state_values = self.actor_critic(batch_states)
        action_logprobs = normal_log_prob(action_means, log_std, batch_actions)
        ratios = (action_logprobs - batch_logprobs).exp()
        unclipped = ratios * batch_advantages
        clipped = ratios.clip(1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        actor_loss = -unclipped.minimum(clipped).mean()
        critic_loss = (state_values.squeeze(-1) - batch_returns).square().mean()
        entropy = (log_std + NORMAL_ENTROPY_CONSTANT).sum()
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

    @TinyJit
    @Context(TRAINING=1)
    def train_step(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        self.optimizer.zero_grad()
        loss = self.compute_loss(
            batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
        )
        loss.backward()
        clip_grad_norm(self.actor_parameters, 0.5)
        clip_grad_norm(self.critic_parameters, 0.5)
        return loss.realize(*self.optimizer.schedule_step())

    def update(self, obs, raw_act, rew, done):
        T, N = rew.shape
        obs_np = obs.reshape(-1, state_dim).astype(np.float32) * OBS_SCALE
        raw_act_np = raw_act.reshape(-1, action_dim).astype(np.float32)

        obs_flat = Tensor(obs_np, device=device)
        raw_act_flat = Tensor(raw_act_np, device=device)
        old_mean, old_log_std, old_value = self.actor_critic(obs_flat)
        old_logprobs_np = normal_log_prob(old_mean, old_log_std, raw_act_flat).numpy()
        old_values_np = old_value.numpy().reshape(T, N)
        advantages_np, returns_np = self.compute_advantages(rew, old_values_np, done)

        old_logprobs = Tensor(old_logprobs_np, device=device)
        advantages = Tensor(advantages_np, device=device)
        returns = Tensor(returns_np, device=device)
        num_samples = obs_flat.shape[0]

        for _ in range(self.K_epochs):
            permutation = np.random.permutation(num_samples).astype(np.int32)
            for start in range(0, num_samples, self.batch_size):
                idx = Tensor(permutation[start:start + self.batch_size], device=device)
                batch_states = obs_flat[idx].contiguous().realize()
                batch_actions = raw_act_flat[idx].contiguous().realize()
                batch_logprobs = old_logprobs[idx].contiguous().realize()
                batch_advantages = advantages[idx].contiguous().realize()
                batch_returns = returns[idx].contiguous().realize()
                self.train_step(
                    batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
                )
        self.actor_critic.refresh_action_std()


def rollout(env, actor_critic, num_steps=None, num_episodes=None, deterministic=False):
    assert (num_steps is None) != (num_episodes is None), 'Specify exactly one: num_steps or num_episodes'

    N = env.num_envs
    states, _ = env.reset()
    ep_returns, ep_rets = [], np.zeros(N)

    collect = num_steps is not None
    if collect:
        T = num_steps // N
        obs = np.empty((T, N, state_dim), dtype=np.float32)
        raw_act = np.empty((T, N, action_dim), dtype=np.float32)
        rew = np.empty((T, N), dtype=np.float32)
        done = np.empty((T, N), dtype=np.float32)

    t = 0
    while True:
        actions, raw_actions = actor_critic.act(states, deterministic=deterministic)
        if collect:
            obs[t], raw_act[t] = states, raw_actions

        states, rewards, terminated, truncated, _ = env.step(actions)
        d = np.logical_or(terminated, truncated)
        if collect:
            rew[t], done[t] = rewards, d

        ep_rets += rewards
        track_episode_returns(d, ep_returns, ep_rets)
        t += 1

        if (collect and t >= T) or (num_episodes and len(ep_returns) >= num_episodes):
            break

    return (obs, raw_act, rew, done, ep_returns) if collect else ep_returns


class TrainingContext:
    def __init__(self):
        self.ac = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers)
        self.ppo = PPO(self.ac, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef)

        self.env = make_env(num_envs)
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
    t0 = time.perf_counter()
    obs, raw_act, rew, done, ep_rets = rollout(ctx.env, ctx.ac, num_steps=steps_per_epoch)
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)

    t0 = time.perf_counter()
    ctx.ppo.update(obs, raw_act, rew, done)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)

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
    for epoch in range(max_epochs):
        if train_one_epoch(epoch, ctx):
            break
    ctx.cleanup()


def evaluate_policy(actor_critic, num_episodes=eval_episodes, render=False, env=None, deterministic=True):
    close_env = env is None
    if env is None:
        env = make_env(1 if render else num_episodes, render)
    ep_rets = rollout(env, actor_critic, num_episodes=num_episodes, deterministic=deterministic)
    if close_env:
        env.close()
    return float(np.mean(ep_rets)) if ep_rets else 0.0


if __name__ == '__main__':
    print(f'Using Tinygrad {device} device')
    train()
