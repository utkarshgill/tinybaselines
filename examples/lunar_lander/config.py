import numpy as np
from gymnasium.vector import VectorWrapper

ENV_NAME = 'LunarLanderContinuous-v3'
# https://gymnasium.farama.org/environments/box2d/lunar_lander/
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)


class SeededReset(VectorWrapper):
    def __init__(self, env, seed, stream):
        super().__init__(env)
        self.seed = int(np.random.SeedSequence([seed, stream]).generate_state(1, dtype=np.uint64)[0])

    def reset(self, *, seed=None, options=None):
        seed = self.seed if seed is None else seed
        self.seed = None
        return super().reset(seed=seed, options=options)
