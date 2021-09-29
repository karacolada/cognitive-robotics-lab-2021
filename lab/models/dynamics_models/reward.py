import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self, state_size, hidden_size=None):
        super().__init__()
        self.state_size = state_size
        self.input_size = sum(state_size.values())
        if hidden_size is None:
            hidden_size = self.input_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        input = torch.cat([state[key] for key in self.state_size.keys()], dim=1)  # concatenate per batch
        # squeeze batch+timesteps
        input = input.view(-1, self.input_size)
        reward = self.linear_relu_stack(input)
        return reward.view(-1)