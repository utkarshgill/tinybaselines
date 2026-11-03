from ctypes import CDLL, c_uint, c_void_p

import numpy as np


class SquaredVectorEnv:
    """Headless native environments. Observations and rewards reuse their buffers."""

    metadata = {'autoreset_mode': 'SameStep'}

    def __init__(self, library, num_envs=4):
        if num_envs <= 0:
            raise ValueError('num_envs must be positive')
        self.num_envs = num_envs
        self._lib = CDLL(str(library))
        self._lib.squared_create.argtypes = [c_void_p] * 4 + [c_uint]
        self._lib.squared_create.restype = c_void_p
        for name in ('puf_reset', 'puf_step', 'squared_destroy'):
            function = getattr(self._lib, name)
            function.argtypes = [c_void_p]
            function.restype = None

        self._observations = np.empty((num_envs, 121), dtype=np.uint8)
        self._actions = np.zeros((num_envs, 2), dtype=np.float32)
        self._rewards = np.zeros(num_envs, dtype=np.float32)
        self._terminals = np.zeros(num_envs, dtype=np.float32)
        self._envs = []
        try:
            for i, seed in enumerate(np.random.SeedSequence().generate_state(num_envs)):
                handle = self._lib.squared_create(
                    self._observations[i].ctypes.data, self._actions[i].ctypes.data,
                    self._rewards[i:i+1].ctypes.data, self._terminals[i:i+1].ctypes.data, int(seed))
                if not handle:
                    raise MemoryError('could not allocate Squared Continuous')
                self._envs.append(handle)
        except BaseException:
            self.close()
            raise

    def reset(self):
        if not self._envs:
            raise RuntimeError('environment is closed')
        for handle in self._envs:
            self._lib.puf_reset(handle)
        self._rewards.fill(0)
        self._terminals.fill(0)
        return self._observations, {}

    def step(self, actions):
        if not self._envs:
            raise RuntimeError('environment is closed')
        self._actions[:] = actions
        for handle in self._envs:
            self._lib.puf_step(handle)
        # The native timeout is a terminal loss, with an immediate reset.
        return self._observations, self._rewards, self._terminals.astype(bool), np.zeros(self.num_envs, bool), {}

    def close(self):
        for handle in self._envs:
            self._lib.squared_destroy(handle)
        self._envs.clear()


class RelativeSquaredVectorEnv(SquaredVectorEnv):
    @staticmethod
    def _observe(board):
        def position(tile):
            return np.column_stack(np.divmod((board == tile).argmax(axis=1), 11))

        agent, target = position(1), position(2)
        return np.concatenate((agent - 5, target - agent), axis=1).astype(np.float32) / 5

    def reset(self):
        board, info = super().reset()
        return self._observe(board), info

    def step(self, actions):
        board, *transition = super().step(actions)
        return self._observe(board), *transition
