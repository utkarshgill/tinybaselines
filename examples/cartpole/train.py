import argparse
from contextlib import closing
from pathlib import Path
from statistics import mean

import gymnasium as gym

from tinybaselines import CategoricalActorCritic, PPO, evaluate, learn


def main():
    parser = argparse.ArgumentParser(description='Train PPO on CartPole-v1.')
    parser.add_argument('--timesteps', type=int, default=40_960)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()

    policy = CategoricalActorCritic(4, 2, 64, 2, 2)
    ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=4,
              eps_clip=0.2, batch_size=256, vf_coef=0.5, entropy_coef=0.01)
    with closing(gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v1')] * 4,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )) as env:
        for step in learn(env, ppo, total_timesteps=args.timesteps, rollout_steps=1024):
            print(f'{step.timesteps} transitions')
        ppo.save(args.checkpoint)
        policy = PPO.load(args.checkpoint).actor_critic
        returns = evaluate(env, policy, num_episodes=10)
        print(f'Mean return over {len(returns)} episodes: {mean(returns):.1f}')


if __name__ == '__main__':
    main()
