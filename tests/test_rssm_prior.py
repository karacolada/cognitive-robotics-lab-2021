import torch
from lab.models.dynamics_models.recurrent_model import RSSMPrior

min_stddev = 0.1
state_size = {"stoch_state":30, "det_state":25}
action_size = 2

prior_model = RSSMPrior(min_stddev, state_size, action_size)

stoch_state = torch.randn(state_size["stoch_state"]).unsqueeze(0)
det_state = torch.randn(state_size["det_state"]).unsqueeze(0)
prev_state = {"stoch_state": stoch_state, "det_state": det_state}
prev_action = torch.randn(action_size).unsqueeze(0)

det_state, mean, stddev = prior_model(prev_state, prev_action)
stoch_state = mean + stddev*torch.randn_like(mean)  # sample from normal distribution
state = {"det_state": det_state, "stoch_state": stoch_state, "mean": mean, "stddev": stddev}

for k in state.keys():
    print("Shape of {}: {}".format(k, state[k].shape))