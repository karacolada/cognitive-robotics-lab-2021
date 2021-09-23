from torch import nn

class VectorDecoder(nn.Module):
    # reconstruct observation from state?
    def __init__(self, observation_size, state_size):
        super(VectorDecoder, self).__init__()
        self.flatten = nn.Flatten
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, observation_size),
            nn.ReLU(),
            nn.Linear(observation_size, observation_size),
            nn.ReLU(),
            nn.Linear(observation_size, observation_size)
        )
    
    def forward(self, state):
        state = self.flatten(state)
        observation = self.linear_relu_stack(state)
        return observation

class ImageDecoder(nn.Module):
    pass