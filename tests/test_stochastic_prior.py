import torch
from lab.models.dynamics_models.stochastic_model import StochasticPrior

min_stddev = 0.1
state_size = 30
action_size = 50

prior_model = StochasticPrior(min_stddev, state_size, action_size)

prev_state = torch.randn(state_size)
prev_action = torch.randn(action_size)

mean, stddev = prior_model(prev_state, prev_action)
state = torch.normal(mean, stddev)

print(state.shape) # should be equal to state_size