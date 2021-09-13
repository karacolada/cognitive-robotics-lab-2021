from lab.environment.env import Env
from lab.environment.wrapper import EnvWrapper


class TimeLimit(EnvWrapper):
    def __init__(self, env: Env, time_limit: int, action_repeat: int):
        super().__init__(env)
        self._time_limit = time_limit
        self._action_repeat = action_repeat
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += self._action_repeat
        if self._step >= self._time_limit:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()
