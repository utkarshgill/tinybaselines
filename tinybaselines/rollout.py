from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class RolloutState:
    observations: np.ndarray
    episode_returns: np.ndarray


class RolloutBatch(NamedTuple):
    observations: np.ndarray
    raw_actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_observations: np.ndarray
