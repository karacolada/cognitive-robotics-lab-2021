import torch
import torch.nn as nn
from lab.models.baselines.networks import mlp
import torch.nn.functional as F


class MLPValueFunction(nn.Module):
    def __init__(self, observation_size, action_size, hidden_sizes, activation):
        super().__init__()
        q_sizes = [observation_size + action_size] + list(hidden_sizes) + [1]
        self.q = mlp(q_sizes, activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPDiscreteValueFunction(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_sizes, activation):
        super().__init__()
        q_sizes = [observation_size] + list(hidden_sizes) + [num_actions]
        self.q = mlp(q_sizes, activation)

    def forward(self, obs):
        q = self.q(obs)
        return q


class ConvDQNValueFunction(nn.Module):
    def __init__(self, h, w, outputs):
        super(ConvDQNValueFunction, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
