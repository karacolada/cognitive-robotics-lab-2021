import torch
from lab.models.dynamics_models.recurrent_model import RSSMPosterior

min_stddev = 0.1
state_size = {"stoch_state":30, "det_state":25}
embedded_size = 15

posterior_model = RSSMPosterior(min_stddev, state_size, embedded_size)

stoch_state = torch.randn(state_size["stoch_state"]).unsqueeze(0)
det_state = torch.randn(state_size["det_state"]).unsqueeze(0)
prev_state = {"stoch_state": stoch_state, "det_state": det_state}
emb_observation = torch.randn(embedded_size).unsqueeze(0)

post_mean, post_stddev = posterior_model(prev_state["det_state"], emb_observation)
post_state = post_mean + post_stddev*torch.randn_like(post_mean)  # sample from normal distribution
state = {"det_state": prev_state["det_state"], "stoch_state": post_state, "mean": post_mean, "stddev": post_stddev}

for k in state.keys():
    print("Shape of {}: {}".format(k, state[k].shape))