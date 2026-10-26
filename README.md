# tinybaselines

PPO in tinygrad. Continuous actions clipped to `[-1, 1]`.

LunarLander needs Python 3.11 and clang.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r examples/lunar_lander/requirements.txt
```

Collect transitions, update, repeat.

```python
import numpy as np
from tinybaselines import GaussianActorCritic, PPO, RolloutState, rollout
from examples.lunar_lander.config import OBS_SCALE
from examples.lunar_lander.train import make_env

env = make_env(4, asynchronous=False)
policy = GaussianActorCritic(state_dim=8, action_dim=2, hidden_dim=128,
                            actor_layers=4, critic_layers=4, observation_scale=OBS_SCALE)
ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95,
          K_epochs=20, eps_clip=0.2, batch_size=10_000, vf_coef=1.0, entropy_coef=0.001)
state = RolloutState(env.reset()[0], np.zeros(env.num_envs))

for _ in range(100):
    batch, returns = rollout(env, policy, num_steps=20_000, state=state)
    ppo.update(batch)
env.close()

policy.save('policy.safetensors')
policy = GaussianActorCritic.load('policy.safetensors')
actions, raw_actions = policy.act(state.observations, deterministic=True)
```

`num_steps` counts transitions across all environments. Use a multiple of
`env.num_envs`. Keep the same `state` between batches so unfinished episodes
continue. `returns` contains completed episode returns. `actions` go to the
environment; `raw_actions` go into PPO.
The collector uses Gymnasium vector environments with same-step autoreset.
Truncations need `infos['final_obs']` from before the reset. True terminals do not.

The full LunarLander trainer, evaluation, and PyTorch reference live in
`examples/lunar_lander`.

```sh
python -m examples.lunar_lander.train
```
