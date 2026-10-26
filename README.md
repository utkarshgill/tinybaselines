# tinybaselines

PPO in tinygrad. Continuous actions clipped to `[-1, 1]`.

LunarLander needs Python 3.11 and clang.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r examples/lunar_lander/requirements.txt
```

Train, then save the policy.

```python
from tinybaselines import GaussianActorCritic, PPO, learn
from examples.lunar_lander.config import OBS_SCALE
from examples.lunar_lander.train import make_env

env = make_env(4, asynchronous=False)
policy = GaussianActorCritic(state_dim=8, action_dim=2, hidden_dim=128,
                            actor_layers=4, critic_layers=4, observation_scale=OBS_SCALE)
ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95,
          K_epochs=20, eps_clip=0.2, batch_size=10_000, vf_coef=1.0, entropy_coef=0.001)

for step in learn(env, ppo, total_timesteps=2_000_000, rollout_steps=20_000):
    print(step.timesteps, step.rollout_seconds, step.update_seconds)

policy.save('policy.safetensors')
policy = GaussianActorCritic.load('policy.safetensors')
observations, _ = env.reset()
actions, raw_actions = policy.act(observations, deterministic=True)
env.close()
```

`total_timesteps` and `rollout_steps` count transitions across all environments.
Use a positive multiple of `env.num_envs` for `rollout_steps`. Training stops
after the first full rollout that reaches the budget. Each step contains completed
`episode_returns` and collection/update times. The caller owns the environment.
Use a separate environment for evaluation during training.
`actions` go to the environment; `raw_actions` go into PPO.
The collector uses Gymnasium vector environments with same-step autoreset.
Truncations need `infos['final_obs']` from before the reset. True terminals do not.

The full LunarLander trainer, evaluation, and PyTorch reference live in
`examples/lunar_lander`.

```sh
python -m examples.lunar_lander.train
```
