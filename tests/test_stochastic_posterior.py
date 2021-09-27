import torch
from lab.models.dynamics_models.stochastic_model import StochasticPrior, StochasticPosterior

min_stddev = 0.1
state_size = 30
action_size = 50
embedded_size = 15

prior_model = StochasticPrior(min_stddev, state_size, action_size)
posterior_model = StochasticPosterior(min_stddev, state_size, embedded_size)

prev_state = torch.randn(state_size)
prev_action = torch.randn(action_size)
emb_observation = torch.randn(embedded_size)

# code from SSM._posterior
prior_mean, prior_stddev = prior_model(prev_state, prev_action)
post_mean, post_stddev = posterior_model(prior_mean, prior_stddev, emb_observation)
state = torch.normal(post_mean, post_stddev)

print(state.shape) # should be equal to state_size