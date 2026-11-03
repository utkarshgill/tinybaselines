import sys
from contextlib import closing

import numpy as np

from tinybaselines import GaussianActorCritic, PPO, learn
from .env import SquaredVectorEnv


def main(library):
    with closing(SquaredVectorEnv(library)) as env:
        observations, _ = env.reset()
        observations[observations == 2] = 0
        observations[:, 61] = 2  # One tile right of the centered agent.
        actions = np.zeros((4, 2), dtype=np.float32)
        actions[:2, 1] = 1
        observations, rewards, terminated, truncated, _ = env.step(actions)
        np.testing.assert_array_equal(rewards, [1, 1, 0, 0])
        np.testing.assert_array_equal(terminated, [True, True, False, False])
        assert not truncated.any() and np.all(observations[2:, 61] == 2)
        assert np.all(observations[:, 60] == 1)
        env.reset()
        actions.fill(0)
        for _ in range(33):
            _, rewards, terminated, truncated, _ = env.step(actions)
            assert np.all(rewards == 0) and not terminated.any() and not truncated.any()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert np.all(rewards == -1) and terminated.all() and not truncated.any()
        assert np.all(observations[:, 60] == 1)
        print('Goal and timeout rewards and terminal flags survive same-step resets.')

        policy = GaussianActorCritic(121, 2, 16, 1, 1)
        ppo = PPO(policy, pi_lr=1e-3, vf_lr=1e-3, gamma=0.99, lamda=0.95, K_epochs=1,
                  eps_clip=0.2, batch_size=256, vf_coef=0.5, entropy_coef=0.01)
        before = policy.log_std.numpy().copy()
        step, = learn(env, ppo, total_timesteps=256, rollout_steps=256)
        assert step.timesteps == 256 and step.episode_returns
        assert np.isfinite(policy.log_std.numpy()).all() and not np.array_equal(policy.log_std.numpy(), before)
        print(f'One PPO update on {step.timesteps} transitions, {len(step.episode_returns)} finished episodes.')


if __name__ == '__main__':
    main(sys.argv[1])
