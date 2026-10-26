from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutState:
    observations: np.ndarray
    episode_returns: np.ndarray
