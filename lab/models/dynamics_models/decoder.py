import torch
from torch import nn

def Decoder(type, observation_size, state_size, hidden_size=None):
    if type == "vector":
        return VectorDecoder(observation_size, state_size, hidden_size)
    elif type == "image":
        return ImageDecoder(state_size)
    else:
        raise ValueError("Type must be image or vector.")

class VectorDecoder(nn.Module):
    # reconstruct observation from state
    def __init__(self, observation_size, state_size, hidden_size=None):
        super(VectorDecoder, self).__init__()
        if hidden_size is None:
            hidden_size = observation_size
        self.state_size = state_size
        self.input_size = sum(state_size.values())
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, observation_size)
        )
    
    def forward(self, state):
        input = torch.cat([state[key] for key in self.state_size.keys()], dim=1)  # concatenate per batch
        # squeeze batch+timesteps
        input = input.view(-1, self.input_size)
        observation = self.linear_relu_stack(input)
        return observation

class ImageDecoder(nn.Module):
    def __init__(self, state_size):
        super(ImageDecoder, self).__init__()
        self.state_size = state_size
        self.input_size = sum(state_size.values())
        self.prep = nn.Linear(self.input_size, 256)
        self.deconv_relu_stack = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
        )
    
    def forward(self, state):
        input = torch.cat([state[key] for key in self.state_size.keys()], dim=1)  # concatenate per batch
        # squeeze batch+timesteps
        input = input.view(-1, self.input_size)
        hidden = self.prep(input)
        hidden = hidden.view(-1, hidden.shape[1], 1, 1)
        observation = self.deconv_relu_stack(hidden)
        return observation