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
    pass