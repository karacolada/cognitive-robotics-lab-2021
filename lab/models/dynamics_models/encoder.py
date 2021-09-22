from torch import nn

class Encoder(nn.Module):
    # learn to embed observation
    def __init__(self, observation_size, embedded_size):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten
        # should it be Linear? Is ReLU ok?
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