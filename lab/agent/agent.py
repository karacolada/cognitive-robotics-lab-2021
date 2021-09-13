from abc import ABC, abstractmethod
import torch
from typing import Iterator, List, Dict, Union, Any, Tuple, Callable
from argparse import ArgumentParser, Namespace
import numpy as np
import torch.nn as nn
from torch import Tensor
from lab.agent.properties import AgentProperties
from lab.agent.optimizers import AgentOptimizersMixin
from lab.replay_buffer.replay_buffer import ReplayBuffer
from lab.trainer.logger import Logger


class Agent(nn.Module, AgentProperties, AgentOptimizersMixin):
    def __init__(self, *args, model_based: bool = False) -> None:
        super(Agent, self).__init__()
        self.save__init__args(locals())
        self.model_based = model_based
        self.replay_buffer = None

    @abstractmethod
    def initialize(self, env_spaces: Dict) -> None:
        """Initializes the networks of the agent based on the specific environment dimensions. Is automatically
        called after __init__()"""
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """Choose one or more optimizers and learning rate schedulers used to optimize the agent."""
        raise NotImplementedError

    @abstractmethod
    def select_action(self, observation: np.ndarray, goal: np.ndarray, explore: bool, episode_start: bool) -> np.ndarray:
        """Computes an action based on the observation given to the agent."""
        raise NotImplementedError

    @abstractmethod
    def learn_on_batch(self, batch) -> Dict:
        """Samples a batch of previous experiences and trains the agent on it. Finally, losses are returned."""
        raise NotImplementedError

    @abstractmethod
    def configure_sampling_args(self):
        raise NotImplementedError

    def sample(self) -> Tuple:
        """Samples a batch of transitions (or sequences of transitions) from the replay buffer."""
        return self.replay_buffer.sample(**self.configure_sampling_args())

    def log(self, name: str, value: Tensor, prefix: str = None) -> None:
        """Adds scalars, images or videos to the logger."""
        if self.time_to_log:
            self.logger.add(name, value, self.train_episodes, prefix, run_id=self.run_id)

    def load(self, path_to_ckpt: str) -> None:
        """Load trained agent from checkpoint."""
        checkpoint = torch.load(path_to_ckpt)
        self.load_state_dict(checkpoint)

    @property
    def train_episodes(self) -> int:
        return self.logger.train_episodes(self.run_id)

    def store_transition(self, observation, goal, action, reward, done, next_observation, next_goal) -> None:
        raise NotImplementedError

    def store_episode(self, observations, actions, rewards, dones, next_observations) -> None:
        raise NotImplementedError

    def connect_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        self.replay_buffer = replay_buffer

    def connect_logger(self, logger: Logger, run_id: int) -> None:
        self.logger = logger
        self.run_id = run_id

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.to(device)
        self.replay_buffer.to(device)

    @torch.no_grad()
    def get_no_grad_action(self, observation: np.ndarray, goal: np.ndarray, explore: bool, episode_start: bool) \
            -> np.ndarray:
        return self.select_action(observation, goal, explore, episode_start)
