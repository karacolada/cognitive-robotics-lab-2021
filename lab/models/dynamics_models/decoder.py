from torch import nn

class VectorDecoder(nn.Module):
    # reconstruct observation from state
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
    def __init__(self, state_size):
        super(ImageDecoder, self).__init__()
        self.prep = nn.Linear(state_size, 256)
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
        hidden = self.prep(state)
        hidden = hidden.view(-1, 256, 1, 1)
        observation = self.deconv_relu_stack(hidden)
        return observation