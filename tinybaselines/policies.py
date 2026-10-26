import json

import numpy as np
from tinygrad import Tensor, TinyJit, nn


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

    def __call__(self, state):
        action_mean = self.actor(state)
        log_std = self.log_std.clip(-5, 2)
        value = self.critic(state)
        return action_mean, log_std, value

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
