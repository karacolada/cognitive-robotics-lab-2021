from lab import Agent
from lab.models.baselines.actor_model import MLPActor
from lab.models.baselines.value_model import MLPValueFunction
from copy import deepcopy
from torch import optim, Tensor
from typing import Dict, Tuple
import torch.nn as nn
import torch
import numpy as np
from lab.replay_buffer.ndarray_tuple import ndarray_tuple


class DDPG(Agent):
    def __init__(self, action_noise: float, batch_size: int, criterion: str, gamma: float, gradient_clip_val: float,
                 optimizer: Dict, polyak: float) -> None:
        super(DDPG, self).__init__(model_based=False)
        self.action_noise = action_noise
        self.batch_size = batch_size
        self.configure_criterion(criterion)
        self.gamma = gamma
        self.gradient_clip_val = gradient_clip_val
        self.lr_scheduler = optimizer["lr_scheduler"]
        self.optimizer = optimizer["type"]
        self.pi_learning_rate = optimizer["pi_learning_rate"]
        self.polyak = polyak
        self.q_learning_rate = optimizer["q_learning_rate"]
        self.make_transition = ndarray_tuple("Transition", ["observation", "goal", "action", "reward", "done",
                                                            "next_observation", "next_goal"])

    def initialize(self, env_spaces: Dict) -> None:
        assert not env_spaces["image_based"], "DDPG can only be run on symbolic environments and not from pixel " \
                                             "observations."
        assert env_spaces["has_continuous_actions"], "Cannot use DDPG on environment with discrete actions."
        observation_size = env_spaces["observation_size"] + env_spaces["goal_size"] if \
            env_spaces["is_goal_conditioned"] else env_spaces["observation_size"]
        self.pi = MLPActor(observation_size, env_spaces["action_size"], (-1., 1.),
                           (256, 256), nn.ReLU)
        self.q = MLPValueFunction(observation_size, env_spaces["action_size"], (256, 256), nn.ReLU)
        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)

    def configure_optimizers(self) -> None:
        self._configure_optimizers([self.optimizer, self.optimizer], [self.pi_learning_rate, self.q_learning_rate],
                                  [self.pi.parameters(), self.q.parameters()], [self.lr_scheduler, self.lr_scheduler])
        self.pi_optimizer, self.q_optimizer = self.optimizers

    def configure_sampling_args(self):
        return {"batch_size": self.batch_size}

    def select_action(self, observation: np.ndarray, goal: np.ndarray, explore: bool, episode_start: bool) \
            -> np.ndarray:
        policy_inputs = np.concatenate([observation, goal], axis=-1) if self.env_spaces["is_goal_conditioned"] \
            else observation
        action = self.pi(torch.as_tensor(policy_inputs, dtype=torch.float32).to(self.device)).cpu().numpy()
        if explore:
            action += self.action_noise * np.random.randn(self.env_spaces["action_size"])
        return np.clip(action, -1., 1.)

    def learn_on_batch(self, batch) -> Dict:
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self._compute_loss_q(batch)
        loss_q.backward()
        if self.gradient_clip_val is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.gradient_clip_val, norm_type=2)
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort computing gradients for it during the policy
        # learning step.
        for p in self.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self._compute_loss_pi(batch)
        loss_pi.backward()
        if self.gradient_clip_val is not None:
            nn.utils.clip_grad_norm_(self.pi.parameters(), self.gradient_clip_val, norm_type=2)
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update(self.pi_target, self.pi, self.polyak)
        self.polyak_update(self.q_target, self.q, self.polyak)

        losses = {"actor_loss": loss_pi.item(), "value_loss": loss_q.item()}
        return losses

    def _compute_loss_q(self, batch: Tuple) -> Tensor:
        observations, goals, actions, rewards, dones, next_observations, next_goals = batch
        inputs = torch.cat([observations, goals], dim=-1) if self.env_spaces["is_goal_conditioned"] else observations
        q = self.q(inputs, actions)

        # Get temporal difference target
        with torch.no_grad():
            next_inputs = torch.cat([next_observations, next_goals], dim=-1) if self.env_spaces["is_goal_conditioned"] \
                else next_observations
            q_pi_target = self.q_target(next_inputs, self.pi_target(next_inputs))
            d = dones.type(torch.uint8)
            td_backup = rewards + self.gamma * (1 - d) * q_pi_target

        # Compute loss
        loss_q = self.criterion(q, td_backup)
        return loss_q

    def _compute_loss_pi(self, batch: Tuple) -> torch.Tensor:
        observations, goals, actions, rewards, dones, next_observations, next_goals = batch
        inputs = torch.cat([observations, goals], dim=-1) if self.env_spaces["is_goal_conditioned"] else observations
        q_pi = self.q(inputs, self.pi(inputs))
        loss_pi = -q_pi.mean()
        return loss_pi

    def store_transition(self, observation, goal, action, reward, done, next_observation, next_goal) -> None:
        transition = self.make_transition(observation=observation, goal=goal, action=action, reward=reward, done=done,
                                          next_observation=next_observation, next_goal=next_goal)
        self.replay_buffer.store_transition(transition)
