import numpy as np
from tinygrad import Context, Tensor, TinyJit, nn

LOG_2PI = float(np.log(2.0 * np.pi))
NORMAL_ENTROPY_CONSTANT = float(0.5 * np.log(2.0 * np.pi * np.e))


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
        self.device = actor_critic.log_std.device
        self.actor_parameters = nn.state.get_parameters(actor_critic.actor) + [actor_critic.log_std]
        self.critic_parameters = nn.state.get_parameters(actor_critic.critic)
        self.pi_optimizer = nn.optim.Adam(self.actor_parameters, lr=pi_lr)
        self.vf_optimizer = nn.optim.Adam(self.critic_parameters, lr=vf_lr)
        self.optimizer = nn.optim.OptimizerGroup(self.pi_optimizer, self.vf_optimizer)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size
        self.vf_coef, self.entropy_coef = vf_coef, entropy_coef
        self._train_step = TinyJit(self._train_step)

    def compute_advantages(self, rewards, state_values, terminated, truncated, next_values):
        T, N = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        episode_ends = terminated | truncated
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * ~terminated[t] - state_values[t]
            gae = delta + self.gamma * self.lamda * ~episode_ends[t] * gae
            advantages[t] = gae
        returns = advantages + state_values
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

    def train_step(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        loss = self._train_step(batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns)
        self.actor_critic.refresh_inference()
        return loss

    @Context(TRAINING=1)
    def _train_step(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        self.optimizer.zero_grad()
        loss = self.compute_loss(
            batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
        )
        loss.backward()
        clip_grad_norm(self.actor_parameters, 0.5)
        clip_grad_norm(self.critic_parameters, 0.5)
        return loss.realize(*self.optimizer.schedule_step())

    def prepare(self, batch):
        T, N = batch.rewards.shape
        obs_flat = self.actor_critic.preprocess(batch.observations.reshape(T * N, -1))
        raw_act_np = batch.raw_actions.reshape(T * N, -1).astype(np.float32)
        raw_act_flat = Tensor(raw_act_np, device=self.device)
        old_mean, old_log_std, old_value = self.actor_critic(obs_flat)
        old_logprobs_np = normal_log_prob(old_mean, old_log_std, raw_act_flat).numpy()
        old_values_np = old_value.numpy().reshape(T, N)
        next_obs_t = self.actor_critic.preprocess(batch.next_observations.reshape(T * N, -1))
        next_values_np = self.actor_critic.critic(next_obs_t).numpy().reshape(T, N)
        advantages_np, returns_np = self.compute_advantages(
            batch.rewards, old_values_np, batch.terminated, batch.truncated, next_values_np
        )

        old_logprobs = Tensor(old_logprobs_np, device=self.device)
        advantages = Tensor(advantages_np, device=self.device)
        returns = Tensor(returns_np, device=self.device)
        return obs_flat, raw_act_flat, old_logprobs, advantages, returns

    def update(self, batch):
        obs_flat, raw_act_flat, old_logprobs, advantages, returns = self.prepare(batch)
        num_samples = obs_flat.shape[0]

        for _ in range(self.K_epochs):
            permutation = np.random.permutation(num_samples).astype(np.int32)
            for start in range(0, num_samples, self.batch_size):
                idx = Tensor(permutation[start:start + self.batch_size], device=self.device)
                batch_states = obs_flat[idx].contiguous().realize()
                batch_actions = raw_act_flat[idx].contiguous().realize()
                batch_logprobs = old_logprobs[idx].contiguous().realize()
                batch_advantages = advantages[idx].contiguous().realize()
                batch_returns = returns[idx].contiguous().realize()
                self.train_step(
                    batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
                )
