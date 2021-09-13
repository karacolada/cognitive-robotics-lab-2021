import torch.nn as nn
from lab.models.baselines.networks import mlp


class MLPActor(nn.Module):
    def __init__(self, observation_size, action_size, action_range, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [observation_size] + list(hidden_sizes) + [action_size]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = action_range[1]

    def forward(self, obs):
        return self.act_limit * self.pi(obs)