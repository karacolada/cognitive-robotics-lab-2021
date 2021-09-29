import torch
from torch import nn
from lab.models.dynamics_models.latent_dynamics_model import LatentDynamicsModel
from lab.models.dynamics_models.encoder import Encoder
from lab.models.dynamics_models.decoder import Decoder


class StochasticModel(LatentDynamicsModel):
    def __init__(self, type, min_stddev, state_size, action_size, observation_size, embedded_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(type, observation_size, embedded_size)
        self.decoder = Decoder(type, observation_size, state_size, hidden_size)
        self.prior_model = StochasticPrior(min_stddev, state_size, action_size, hidden_size)
        self.posterior_model = StochasticPosterior(min_stddev, state_size, embedded_size, hidden_size)

    def _enc(self, observation):
        """e_t = enc(o_t)"""
        return self.encoder(observation)

    def _prior(self, prev_state, prev_action):
        """s_t ~ p(s_t | s_t-1, a_t-1)"""
        mean, stddev = self.prior_model(prev_state["stoch_state"], prev_action)
        state = mean + stddev*torch.randn_like(mean)  # sample from normal distribution
        return {"stoch_state": state, "mean": mean, "stddev": stddev}

    # possibly refactor the bit about priors?
    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | s_t-1, a_t-1, e_t)"""
        prior_mean, prior_stddev = self.prior_model(prev_state["stoch_state"], prev_action)
        post_mean, post_stddev = self.posterior_model(prior_mean, prior_stddev, emb_observation)
        state = post_mean + post_stddev*torch.randn_like(post_mean)  # sample from normal distribution
        return {"stoch_state": state, "mean": post_mean, "stddev": post_stddev}

    def dec(self, state):
        """o_t ~ p(o_t | s_t)"""
        return self.decoder(state)

class StochasticPrior(nn.Module):
    def __init__(self, min_stddev, state_size, action_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = 2*state_size["stoch_state"]
        self.input_size = state_size["stoch_state"] + action_size
        self.min_stddev = min_stddev
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*state_size["stoch_state"])
        )
    
    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        output = self.linear_relu_stack(input)
        mean, stddev = output.chunk(2, dim=-1)
        stddev = nn.functional.softplus(stddev) + self.min_stddev
        return mean, stddev

class StochasticPosterior(nn.Module):
    def __init__(self, min_stddev, state_size, embedded_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = 2*state_size["stoch_state"]
        self.input_size = 2*state_size["stoch_state"] + embedded_size
        self.min_stddev = min_stddev
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*state_size["stoch_state"])
        )

    def forward(self, prior_mean, prior_stddev, embedded):
        input = torch.cat((prior_mean, prior_stddev, embedded), dim=1)
        output = self.linear_relu_stack(input)
        mean, stddev = output.chunk(2, dim=-1)
        stddev = nn.functional.softplus(stddev) + self.min_stddev
        return mean, stddev