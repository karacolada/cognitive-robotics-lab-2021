import cv2
import numpy as np
import torch
from typing import Tuple
from lab.environment.properties import EnvProperties
from lab.environment.wrapper import EnvWrapper


class Env(EnvProperties, EnvWrapper):
    """Base class that should wrap gym, dm-control envs etc."""
    def __init__(self, env: str, show: bool) -> None:
        self._show = show
        self.env_name = env
        self.env_type = self._determine_env_type()
        self._connect_display()
        self.env = self.make()
        super(Env, self).__init__(self.env)

    def _determine_env_type(self):
        try:
            import gym
            _ = gym.make(self.env_name)
            env_type = "gym"
            if self.env_name in self.multiworld_envs:
                env_type = "multiworld"
        except gym.error.UnregisteredEnv:
            import multiworld
            multiworld.register_all_envs()
            try:
                _ = gym.make(self.env_name)
                env_type = "multiworld"
            except gym.error.UnregisteredEnv:
                raise NotImplementedError

        except gym.error.Error:
            try:
                from dm_control import suite
                domain, task = self.env_name.split('-')
                _ = suite.load(domain_name=domain, task_name=task)
                env_type = "dm_control"
            except ValueError:
                raise NotImplementedError
        return env_type

    def make(self):
        if self.is_gym_env or self.is_multiworld_env:
            import gym
            return gym.make(self.env_name)
        elif self.is_dm_control_env:
            from dm_control import suite
            domain, task = self.env_name.split('-')
            return suite.load(domain_name=domain, task_name=task)
        else:
            raise NotImplementedError

    def render(self, mode: str, dims: Tuple = None):
        if self.is_gym_env or self.is_multiworld_env:
            image = self.env.render(mode=mode)
        elif self.is_dm_control_env:
            image = self.env.physics.render(camera_id=0)
        else:
            raise NotImplementedError
        if dims:
            image = torch.tensor(cv2.resize(image, dims, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))
        return image

    def step(self, action):
        if self._show:
            self.show()
        if self.is_gym_env or self.is_multiworld_env:
            return self.env.step(action)
        elif self.is_dm_control_env:
            ts = self.env.step(action)
            return self._obs_dmc2gym(ts.observation), ts.reward, ts.last(), {}
        else:
            raise NotImplementedError

    def reset(self):
        if self.is_gym_env or self.is_multiworld_env:
            return self.env.reset()
        elif self.is_dm_control_env:
            ts = self.env.reset()
            return self._obs_dmc2gym(ts.observation)

    def show(self):
        if self.is_gym_env or self.is_multiworld_env:
            self.env.render()
        elif self.is_dm_control_env:
            cv2.imshow('screen', self.env.physics.render(camera_id=0))
            cv2.waitKey(1)
        else:
            raise NotImplementedError

    def close(self):
        if self.is_dm_control_env:
            cv2.destroyAllWindows()
        self.env.close()

    def sample_random_action(self):
        if self.is_gym_env or self.is_multiworld_env or self.is_dm_control_env:
            if self.has_discrete_actions:
                return np.random.randint(self.gym_env.action_space.n)
            else:
                low = self.action_space.low
                high = self.action_space.high
                return np.random.uniform(low=low, high=high, size=low.shape)
        else:
            raise NotImplementedError

    def seed(self, seed: int) -> None:
        if self.is_gym_env or self.is_multiworld_env:
            self.env.seed(seed)
        elif self.is_dm_control_env:
            from dm_control import suite
            domain, task = self.env_name.split('-')
            self.env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        else:
            raise NotImplementedError
