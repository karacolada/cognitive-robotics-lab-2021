import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Dict, Tuple


class CEMPlanner(nn.Module):
    def __init__(self, action_size: int, planning_horizon: int, optimization_iters: int, candidates: int,
                 top_candidates: int, transition_fn: Callable, objective_fn: Callable, min_action: float = -1.,
                 max_action: float = 1.) -> None:
        super(CEMPlanner, self).__init__()
        self.transition_fn, self.objective_fn = transition_fn, objective_fn
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimization_iters = optimization_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    def forward(self, state: Dict) -> Tensor:
        """Plans via cross entropy method by rolling out future trajectories for sampled actions."""
        self._init_batch_size_and_device(state)
        init_state = self._expand_state_by_candidates(state)
        action_mean, action_std_dev = self._init_action_sequence_params()  # q(a_t:t+H) ~ N(0, I)

        for i in range(self.optimization_iters):
            actions = self._sample_action_sequences(action_mean, action_std_dev, self.batch_size)
            prev_state = init_state
            returns = torch.zeros(self.batch_size * self.candidates).type_as(actions)
            for t in range(self.planning_horizon):
                state = self.transition_fn(prev_state, actions[t])
                reward = self.objective_fn(state["state"])
                returns += reward
                prev_state = state
            action_mean, action_std_dev = self._refit_belief_to_topk(returns, actions, self.batch_size)
        return action_mean[0].squeeze(dim=1)  # Return first action mean µ_t

    def _expand_state_by_candidates(self, state: Dict) -> Dict:
        expanded_state = {"state": {}}
        for key in state["state"].keys():
            batch_size, state_size = state["state"][key].size(0), state["state"][key].size(1)
            expanded_state["state"][key] = state["state"][key].unsqueeze(dim=1).expand(batch_size, self.candidates,
                                                                                       state_size)
            expanded_state["state"][key] = expanded_state["state"][key].reshape(-1, state_size)
        return expanded_state

    def _init_action_sequence_params(self) -> Tuple:
        action_mean = torch.zeros(self.planning_horizon, self.batch_size, 1, self.action_size)
        action_std_dev = torch.ones(self.planning_horizon, self.batch_size, 1, self.action_size)
        action_mean = action_mean.to(self.device)
        action_std_dev = action_std_dev.to(self.device)
        return action_mean, action_std_dev

    def _sample_action_sequences(self, action_mean: Tensor, action_std_dev: Tensor, batch_size: int) -> Tensor:
        sample = torch.randn(self.planning_horizon, batch_size, self.candidates, self.action_size, device=action_mean.device)
        actions = action_mean + action_std_dev * sample
        actions = actions.view(self.planning_horizon, batch_size * self.candidates, self.action_size)  # Merge batch_size and candidates.
        return actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range

    def _refit_belief_to_topk(self, returns: Tensor, actions: Tensor, batch_size: int) -> Tuple:
        returns = returns.reshape(batch_size, self.candidates)
        _, topk = returns.topk(self.top_candidates, dim=1, largest=True, sorted=False)

        topk += self.candidates * torch.arange(0, batch_size, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
        best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, batch_size, self.top_candidates,
                                                         self.action_size)
        # Update belief with new means and standard deviations
        action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False,
                                                                                               keepdim=True)
        return action_mean, action_std_dev

    def _init_batch_size_and_device(self, state: Dict) -> None:
        self.batch_size = next(iter(state["state"].values())).size(0)
        self.device = next(iter(state["state"].values())).device
