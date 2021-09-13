from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn
from typing import Dict, Optional


class LatentDynamicsModel(ABC, nn.Module):
    def __call__(self, prev_state: Dict, prev_action: Tensor, observation: Optional[Tensor] = None) -> Dict:
        prev_state = prev_state["state"]
        prior = self._prior(prev_state, prev_action)
        if observation is not None:
            emb_observation = self._enc(observation)
            posterior = self._posterior(prev_state, prev_action, emb_observation)
        else:
            posterior = prior
        return {"state": posterior, "out": (prior, posterior)}

    @abstractmethod
    def _enc(self, observation):
        """e_t = enc(o_t)"""
        raise NotImplementedError

    @abstractmethod
    def _prior(self, prev_state, prev_action):
        """s_t ~ p(s_t | s_t-1, a_t-1)"""
        raise NotImplementedError

    @abstractmethod
    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | s_t-1, a_t-1, e_t)"""
        raise NotImplementedError

    @abstractmethod
    def dec(self, state):
        """o_t ~ p(o_t | s_t)"""
        raise NotImplementedError
