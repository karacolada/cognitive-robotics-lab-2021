import torch
from lab.models.dynamics_models.reward import RewardModel

state_size = 30

model = RewardModel(state_size, hidden_size=15)

input = torch.randn(state_size)
input = input.unsqueeze(0)

reward = model(input)
print("Size of reward batch: {}".format(reward.size()))
