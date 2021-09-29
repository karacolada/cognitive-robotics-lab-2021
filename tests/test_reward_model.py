import torch
from lab.models.dynamics_models.reward import RewardModel

state_size_rssm = {"stoch_state":30, "det_state":25}
state_size_rnn = {"det_state":25}
state_size_ssm = {"stoch_state":30}
stoch_state = torch.randn(30).unsqueeze(0)
det_state = torch.randn(25).unsqueeze(0)
state = {"stoch_state":stoch_state, "det_state": det_state}

print("Reward model test: RSSM")

model = RewardModel(state_size_rssm, hidden_size=15)

reward = model(state)
print("Size of reward batch: {}".format(reward.size()))

print("------------------------\nReward model test: RNN")

model = RewardModel(state_size_rnn, hidden_size=15)

reward = model(state)
print("Size of reward batch: {}".format(reward.size()))


print("------------------------\nReward model test: SSM")

model = RewardModel(state_size_ssm, hidden_size=15)

reward = model(state)
print("Size of reward batch: {}".format(reward.size()))