import json

import numpy as np
from tinygrad import Tensor, TinyJit, nn

LOG_2PI = float(np.log(2.0 * np.pi))
NORMAL_ENTROPY_CONSTANT = float(0.5 * np.log(2.0 * np.pi * np.e))


def normal_log_prob(action_mean, log_std, actions):
    normalized = (actions - action_mean) / log_std.exp()
    return (-0.5 * (normalized.square() + 2.0 * log_std + LOG_2PI)).sum(axis=-1)


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dim, layer_count):
        self.layers = [nn.Linear(input_dim, hidden_dim)]
        self.layers.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_count - 1))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)


class GaussianActorCritic:
    _checkpoint_type = 'gaussian'

    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers, observation_scale=None):
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                           actor_layers=actor_layers, critic_layers=critic_layers)
        self.actor = MLP(state_dim, action_dim, hidden_dim, actor_layers)
        self.critic = MLP(state_dim, 1, hidden_dim, critic_layers)
        self.actor.layers[-1].weight.assign(self.actor.layers[-1].weight * 0.01).realize()
        self.actor.layers[-1].bias.assign(Tensor.zeros_like(self.actor.layers[-1].bias)).realize()
        self.log_std = Tensor.full((action_dim,), -0.7).realize()
        scale = np.ones(state_dim, np.float32) if observation_scale is None else np.asarray(observation_scale, np.float32)
        self.observation_scale = Tensor(scale.copy()).is_param_(False).realize()
        self.jit_shape = None
        self.actor_jit = TinyJit(lambda state: self.actor(state).realize())
        self.refresh_inference()

    @property
    def actor_parameters(self):
        return nn.state.get_parameters(self.actor) + [self.log_std]

    @property
    def device(self):
        return self.log_std.device

    def __call__(self, state):
        action_mean = self.actor(state)
        log_std = self.log_std.clip(-5, 2)
        value = self.critic(state)
        return action_mean, log_std, value

    def evaluate_actions(self, states, actions):
        """Return values, action log probabilities, and mean entropy for preprocessed states."""
        means, log_std, values = self(states)
        log_probs = normal_log_prob(means, log_std, actions)
        entropy = (log_std + NORMAL_ENTROPY_CONSTANT).sum()
        return values, log_probs, entropy

    def preprocess(self, observations):
        return Tensor(observations.astype(np.float32), device=self.observation_scale.device) * self.observation_scale

    def act(self, state, deterministic=False):
        state_tensor = self.preprocess(state)
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

    def state_dict(self):
        return nn.state.get_state_dict(dict(actor=self.actor, critic=self.critic, log_std=self.log_std,
                                            observation_scale=self.observation_scale))

    def save(self, path):
        nn.state.safe_save(self.state_dict(), path, metadata={'config': json.dumps(self.config)})

    @classmethod
    def load(cls, path):
        _, _, header = nn.state.safe_load_metadata(path)
        model = cls(**json.loads(header['__metadata__']['config']))
        nn.state.load_state_dict(model.state_dict(), nn.state.safe_load(path), verbose=False)
        model.refresh_inference()
        return model

    def refresh_inference(self):
        self.actor_jit.reset()
        self.action_std = np.exp(self.log_std.clip(-5, 2).numpy())
        if not np.isfinite(self.action_std).all():
            raise FloatingPointError('non-finite policy standard deviation')


class CategoricalActorCritic:
    """Actor-critic for a single discrete action per observation."""

    _checkpoint_type = 'categorical'

    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                           actor_layers=actor_layers, critic_layers=critic_layers)
        self.actor = MLP(state_dim, action_dim, hidden_dim, actor_layers)
        self.critic = MLP(state_dim, 1, hidden_dim, critic_layers)
        self.actor_jit = TinyJit(lambda states: self.actor(states).realize())
        self.jit_shape = None
        self._cpu_input = None

    @property
    def actor_parameters(self):
        return nn.state.get_parameters(self.actor)

    @property
    def device(self):
        return self.actor.layers[0].weight.device

    def preprocess(self, observations):
        return Tensor(observations.astype(np.float32), device=self.device)

    def evaluate_actions(self, states, actions):
        log_probs = self.actor(states).log_softmax(axis=-1)
        selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(axis=-1).mean()
        return self.critic(states), selected, entropy

    def _actor_logits(self, state):
        if self.jit_shape != state.shape:
            self.refresh_inference()
            self.jit_shape = state.shape
            self._cpu_input = None
        if self.device == 'CPU':
            if self._cpu_input is None:
                # Keep the NumPy owner alive for the lifetime of the tensor.
                host = np.empty(state.shape, dtype=np.float32)
                self._cpu_input = host, Tensor.from_blob(host.ctypes.data, host.shape, device='CPU', dtype='float32')
            host, states = self._cpu_input
            host[:] = state
        else:
            states = self.preprocess(state)
        return self.actor_jit(states).numpy()

    def act(self, state, deterministic=False):
        logits = self._actor_logits(state)
        scores = logits if deterministic else logits + np.random.gumbel(size=logits.shape)
        actions = scores.argmax(axis=-1).astype(np.int32)
        return actions, actions

    def refresh_inference(self):
        self.actor_jit.reset()

    def state_dict(self):
        return nn.state.get_state_dict(dict(actor=self.actor, critic=self.critic))
