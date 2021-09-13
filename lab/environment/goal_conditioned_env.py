from lab.environment.env import Env
from lab.environment.wrapper import EnvWrapper
from typing import Tuple
import numpy as np


class GoalConditionedEnv(EnvWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, action: np.ndarray) -> Tuple:
        observation, reward, done, info = self.env.step(action)
        if self.env.is_goal_conditioned:
            goal = observation["desired_goal"]
            observation = observation["observation"]
        else:
            goal = np.array([])
        return observation, goal, reward, done, info

    def reset(self):
        observation = self.env.reset()
        if self.env.is_goal_conditioned:
            goal = observation["desired_goal"]
            observation = observation["observation"]
        else:
            goal = np.array([])
        return observation, goal
