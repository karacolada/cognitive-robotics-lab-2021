from torch import nn

class RewardModel(nn.Module):
    def __init__(self, state_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = state_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        reward = self.linear_relu_stack(state)
        return reward