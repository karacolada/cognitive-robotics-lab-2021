from lab.environment.env import Env
from lab.environment.wrapper import EnvWrapper
import numpy as np
from typing import Tuple


class DataType(EnvWrapper):
    def __init__(self, env: Env, dtype=np.float32):
        super().__init__(env)
        self._dtype = dtype

    def step(self, action):
        items = self.env.step(action)
        return self._adjust_dtype(items)

    def reset(self):
        items = self.env.reset()
        return self._adjust_dtype(items)

    def _adjust_dtype(self, items: Tuple) -> Tuple:
        if not isinstance(items, Tuple):
            items = tuple(items)
        processed_items = []
        for item in items:
            if isinstance(item, np.ndarray):
                processed_items.append(item.astype(self._dtype))
            else:
                processed_items.append(item)
        return tuple(processed_items)
