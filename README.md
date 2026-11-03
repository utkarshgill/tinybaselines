# tinybaselines

PPO in tinygrad for discrete and continuous actions. Continuous actions are clipped to `[-1, 1]`.

## LunarLander from PyPI

Use Python 3.11 and clang. The library install does not include Gymnasium.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install tinybaselines==0.1.0a3
```

Install Gymnasium and Box2D to run this example:

```sh
python -m pip install 'gymnasium[box2d]==1.4.0'
```

Save this as `train_lander.py`. It runs one PPO update and evaluates the reloaded checkpoint.

```python
from contextlib import closing
from statistics import mean

import gymnasium as gym

from tinybaselines import GaussianActorCritic, PPO, evaluate, learn

policy = GaussianActorCritic(state_dim=8, action_dim=2, hidden_dim=128,
                             actor_layers=4, critic_layers=4,
                             observation_scale=[10, 6.666, 5, 7.5, 1, 2.5, 1, 1])
ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95,
          K_epochs=20, eps_clip=0.2, batch_size=10_000, vf_coef=1.0, entropy_coef=0.001)

with closing(gym.vector.SyncVectorEnv(
    [lambda: gym.make('LunarLander-v3', continuous=True)] * 4,
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
)) as env:
    for step in learn(env, ppo, total_timesteps=20_000, rollout_steps=20_000):
        print(step.timesteps, step.rollout_seconds, step.update_seconds)
    ppo.save('ppo.safetensors')
    policy = PPO.load('ppo.safetensors').actor_critic
    print(mean(evaluate(env, policy, num_episodes=10)))
```

```sh
DEV=CPU BEAM=0 python train_lander.py
```

`total_timesteps` and `rollout_steps` count transitions across all environments.
Use a positive multiple of `env.num_envs` for `rollout_steps`. Training stops
after the first full rollout that reaches the budget. Each step contains completed
`episode_returns` and collection/update times. The caller owns the environment.
Use a separate environment for evaluation during training.
`evaluate` resets its environment and returns complete episode returns. It samples
actions by default; `deterministic=True` uses clipped policy means.
`actions` go to the environment; `raw_actions` go into PPO.
The collector uses Gymnasium vector environments with same-step autoreset.
For time-limit bootstrapping, the collector reads pre-reset observations from `infos['final_obs']`.

PPO checkpoints contain the policy, Adam state, and PPO settings.
Continue training by passing the loaded PPO to `learn` with a new transition budget.
Each `learn` call starts fresh episodes and a new transition counter.
`GaussianActorCritic.save(path)` exports a Gaussian policy for inference.

## Repository examples

From a checkout, run the full LunarLander trainer:

```sh
pip install -e . -r examples/lunar_lander/requirements.txt
python -m examples.lunar_lander.train
```

CartPole uses `CategoricalActorCritic`. The example trains, saves a checkpoint,
and evaluates the reloaded policy over 10 episodes.

```sh
pip install -e . 'gymnasium==1.4.0'
DEV=CPU BEAM=0 python -m examples.cartpole.train --checkpoint cartpole.safetensors
```

Three independent four-environment CPU runs averaged 500.00, 496.67, and 494.37
after 102,400 transitions each. All cleared 475 over 100 fresh stochastic episodes
per reloaded checkpoint. [Run records](examples/cartpole/results/cpu.jsonl).

Pendulum uses three observations and one action. Its example rescales policy actions
from `[-1, 1]` to torque in `[-2, 2]`.

```sh
pip install -e . -r examples/pendulum/requirements.txt
DEV=CPU BEAM=0 python -m examples.pendulum.train --output runs/pendulum
```

The output directory contains the configuration, episode returns, update times,
and a PPO checkpoint. Initial and final evaluations each use 100 fresh stochastic episodes.

Three independent four-environment CPU runs reached mean returns of -141.4, -162.2,
and -150.8 after 200,704 transitions each. Each final evaluation used the reloaded checkpoint.
Our acceptance cutoff is -200 over 100 fresh stochastic episodes per run; all three passed.
Gymnasium defines no solved threshold for Pendulum. We chose this cutoff after the first run.
Run records: [1](examples/pendulum/results/cpu.jsonl),
[2](examples/pendulum/results/cpu-2.jsonl), [3](examples/pendulum/results/cpu-3.jsonl).
