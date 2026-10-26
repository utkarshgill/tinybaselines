from .rollout import rollout


def evaluate(env, policy, num_episodes, deterministic=False):
    """Return total rewards from num_episodes fresh episodes."""
    return rollout(env, policy, num_episodes=num_episodes, deterministic=deterministic)
