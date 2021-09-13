import numpy as np
from lab.environment.wrapper import EnvWrapper
from lab.environment.env import Env
from typing import Tuple


class NormalizeActions(EnvWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def step(self, action: np.ndarray) -> Tuple:
        unnormalized_action = ((action + 1) * ((self._high - self._low) / 2)) + self._low
        unnormalized_action = np.where(self._mask, unnormalized_action, action)
        return self.env.step(unnormalized_action)
