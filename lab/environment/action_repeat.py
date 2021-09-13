from lab.environment.env import Env
from lab.environment.wrapper import EnvWrapper


class ActionRepeat(EnvWrapper):
    def __init__(self, env: Env, action_repeat: int):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        reward = 0
        for k in range(self.action_repeat):
            observation, reward_k, done, info = self.env.step(action)
            reward += reward_k
        return observation, reward, done, info
