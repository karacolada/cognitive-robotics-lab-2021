from torch import nn

class VectorEncoder(nn.Module):
    # learn to embed observation
    def __init__(self, observation_size, embedded_size):
        super(VectorEncoder, self).__init__()
        self.flatten = nn.Flatten
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(observation_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, embedded_size)
        )
    
    def forward(self, observation):
        observation = self.flatten(observation)
        embedded = self.linear_relu_stack(observation)
        return embedded

class ImageEncoder(nn.Module):
    def __init__(self, embedded_size):
        super(ImageEncoder, self).__init__()
        # assuming input of 64x64
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, embedded_size)
        )
    
    def forward(self, observation):
        embedded = self.conv_relu_stack(observation)
        return embedded