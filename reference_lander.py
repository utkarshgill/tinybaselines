import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import warnings
from tqdm import trange
from examples.lunar_lander.config import ENV_NAME, OBS_SCALE
from rollout import RolloutBatch, RolloutState

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
METAL = bool(int(os.getenv('METAL', '0')))

device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

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
        ma = np.convolve(returns, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(returns)), ma, label='100-ep MA', linewidth=2)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({threshold})')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    plt.pause(0.01)


def track_episode_returns(done_mask, ep_returns, ep_rets, remaining=None):
    for idx in np.where(done_mask)[0]:
        if remaining is None or remaining[idx] > 0:
            ep_returns.append(ep_rets[idx])
            if remaining is not None:
                remaining[idx] -= 1
        ep_rets[idx] = 0.0

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                           actor_layers=actor_layers, critic_layers=critic_layers)
        self.register_buffer('observation_scale', torch.from_numpy(OBS_SCALE.copy()))
        
        actor = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        # small mu init → near-zero thrust initially (slow descent > random crashes)
        self.actor[-1].weight.data.mul_(0.01)
        self.actor[-1].bias.data.zero_()
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        
        critic = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(critic_layers - 1):
            critic.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        critic.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*critic)

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.clamp(-5, 2).exp().expand_as(action_mean)
        value = self.critic(state)
        return action_mean, action_std, value
    
    def preprocess(self, observations):
        return torch.as_tensor(observations, device=self.observation_scale.device, dtype=torch.float32) * self.observation_scale

    @torch.inference_mode()
    def act(self, state, deterministic=False):
        state_tensor = self.preprocess(state)
        action_mean = self.actor(state_tensor)
        action_std = self.log_std.clamp(-5, 2).exp().expand_as(action_mean)
        if deterministic:
            action = action_mean.clamp(-1, 1)
            return action.cpu().numpy(), action_mean.cpu().numpy()
        sample = torch.distributions.Normal(action_mean, action_std).sample()
        action = sample.clamp(-1, 1)
        return action.cpu().numpy(), sample.cpu().numpy()

    def save(self, path):
        torch.save({'config': self.config, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        return model


class PPO:
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.pi_optimizer = optim.Adam(list(actor_critic.actor.parameters()) + [actor_critic.log_std], lr=pi_lr)
        self.vf_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.vf_coef, self.entropy_coef = eps_clip, batch_size, vf_coef, entropy_coef

    def compute_advantages(self, rewards, state_values, terminated, truncated, next_values):
        T, N = rewards.shape
        advantages, gae = torch.zeros_like(rewards), torch.zeros(N, device=rewards.device)
        episode_ends = terminated | truncated
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * ~terminated[t] - state_values[t]
            gae = delta + self.gamma * self.lamda * ~episode_ends[t] * gae
            advantages[t] = gae
        returns = advantages + state_values
        advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_loss(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = dist.log_prob(batch_actions).sum(-1)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().sum(-1).mean()
        
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
    
    def train_step(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        self.pi_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()
        loss = self.compute_loss(batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std], max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=0.5)
        self.pi_optimizer.step()
        self.vf_optimizer.step()
        return loss.detach()

    def prepare(self, batch):
        T, N = batch.rewards.shape
        obs_flat = self.actor_critic.preprocess(batch.observations.reshape(T * N, -1))
        raw_act_flat = torch.from_numpy(batch.raw_actions.reshape(T * N, -1)).to(device=device, dtype=torch.float32)
        rew_t = torch.from_numpy(batch.rewards).to(device=device, dtype=torch.float32)
        terminated_t = torch.from_numpy(batch.terminated).to(device=device)
        truncated_t = torch.from_numpy(batch.truncated).to(device=device)
        
        with torch.no_grad():
            mean, std, val = self.actor_critic(obs_flat)
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = dist.log_prob(raw_act_flat).sum(-1)
            old_values = val.squeeze(-1).view(T, N)
            next_obs_t = self.actor_critic.preprocess(batch.next_observations.reshape(T * N, -1))
            next_values = self.actor_critic.critic(next_obs_t).squeeze(-1).view(T, N)
            advantages, returns = self.compute_advantages(rew_t, old_values, terminated_t, truncated_t, next_values)
        
        return obs_flat, raw_act_flat, old_logprobs, advantages, returns

    def update(self, batch):
        obs_flat, raw_act_flat, old_logprobs, advantages, returns = self.prepare(batch)
        num_samples = obs_flat.size(0)
        for _ in range(self.K_epochs):
            perm = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                self.train_step(obs_flat[idx], raw_act_flat[idx], old_logprobs[idx], advantages[idx], returns[idx])

def rollout(env, actor_critic, num_steps=None, num_episodes=None, deterministic=False, state=None):
    assert (num_steps is None) != (num_episodes is None), "Specify exactly one: num_steps or num_episodes"
    if num_episodes is not None and num_episodes <= 0:
        raise ValueError('num_episodes must be positive')
    
    N = env.num_envs
    if state is None:
        state = RolloutState(env.reset()[0], np.zeros(N))
    states = state.observations
    ep_returns, ep_rets = [], state.episode_returns
    
    collect = num_steps is not None
    remaining = None if collect else (num_episodes + np.arange(N)) // N
    if collect:
        T = num_steps // N
        obs = np.empty((T, N, state_dim), dtype=np.float32)
        next_obs = np.empty_like(obs)
        raw_act = np.empty((T, N, action_dim), dtype=np.float32)
        rew = np.empty((T, N), dtype=np.float32)
        terminations = np.empty((T, N), dtype=bool)
        truncations = np.empty((T, N), dtype=bool)
    
    t = 0
    while True:
        actions, raw_actions = actor_critic.act(states, deterministic=deterministic)
        if collect:
            obs[t], raw_act[t] = states, raw_actions
        
        states, rewards, terminated, truncated, infos = env.step(actions)
        d = np.logical_or(terminated, truncated)
        if collect:
            rew[t], terminations[t], truncations[t] = rewards, terminated, truncated
            next_obs[t] = states
            for idx in np.flatnonzero(d):
                next_obs[t, idx] = infos['final_obs'][idx]
        
        ep_rets += rewards
        track_episode_returns(d, ep_returns, ep_rets, remaining)
        t += 1
        
        if (collect and t >= T) or (not collect and not remaining.any()):
            break
    
    state.observations = states
    return (RolloutBatch(obs, raw_act, rew, terminations, truncations, next_obs), ep_returns) if collect else ep_returns

class TrainingContext:
    def __init__(self):
        self.ac = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers).to(device)
        self.ppo = PPO(self.ac, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef)
        
        self.env = make_env(num_envs)
        self.rollout_state = RolloutState(self.env.reset()[0], np.zeros(num_envs))
        self.eval_env = make_env(min(num_envs, eval_episodes), asynchronous=False)
        self.all_episode_returns = []
        self.last_eval_stochastic = float('-inf')
        self.pbar = trange(max_epochs, desc="Training", unit='epoch')
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
    batch, ep_rets = rollout(
        ctx.env, ctx.ac, num_steps=steps_per_epoch, state=ctx.rollout_state
    )
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    t0 = time.perf_counter()
    ctx.ppo.update(batch)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    ctx.all_episode_returns.extend(ep_rets)
    ctx.pbar.update(1)
    
    if epoch % eval_interval == 0:
        ctx.last_eval_stochastic = evaluate_policy(
            ctx.ac, env=ctx.eval_env, deterministic=False
        )
        if RENDER:
            evaluate_policy(ctx.ac, render=True, num_episodes=RENDER_EPISODES, deterministic=False)
    
    if epoch % log_interval == 0:
        s = ctx.ac.log_std.clamp(-5, 2).exp().detach().cpu().numpy()
        rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
        update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
        total_ms = rollout_ms + update_ms
        ctx.pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_rets):3d}  ret={np.mean(ep_rets):7.1f}±{np.std(ep_rets):5.1f}  eval_stoch={ctx.last_eval_stochastic:6.1f}  σ=[{s[0]:.2f} {s[1]:.2f}]  ⏱ {total_ms:.0f}ms (rollout:{rollout_ms:.0f}ms update:{update_ms:.0f}ms)")
    
    if PLOT and epoch % (log_interval * 2) == 0:
        update_plot(ctx.ax, ctx.all_episode_returns, solved_threshold)
    
    if ctx.last_eval_stochastic >= solved_threshold:
        ctx.pbar.write(f"\n{'='*60}\nSOLVED at epoch {epoch}! eval_stoch={ctx.last_eval_stochastic:.1f} ≥ {solved_threshold}\n{'='*60}")
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
    print(f"Using {device} device")
    if METAL:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        if device.type == 'mps':
            print("✅ MPS device active")
        else:
            print("⚠️  METAL=1 but device is not MPS")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    train()
