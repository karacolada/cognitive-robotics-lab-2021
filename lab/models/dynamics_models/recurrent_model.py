import torch
from torch import nn
from lab.models.dynamics_models.latent_dynamics_model import LatentDynamicsModel
from lab.models.dynamics_models.encoder import Encoder
from lab.models.dynamics_models.decoder import Decoder


class RSSModel(LatentDynamicsModel):
    def __init__(self, type, min_stddev, state_size, action_size, observation_size, embedded_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(type, observation_size, embedded_size)
        self.decoder = Decoder(type, observation_size, state_size, hidden_size)
        self.prior_model = RSSMPrior(min_stddev, state_size, action_size, hidden_size)
        self.posterior_model = None

    def _enc(self, observation):
        """e_t = enc(o_t)"""
        return self.encoder(observation)

    def _prior(self, prev_state, prev_action):
        """ h_t = f(h_t−1 , s_t−1 , a_t−1), s_t ~ p(s_t | h_t)"""
        det_state, mean, stddev = self.prior_model(prev_state, prev_action)
        stoch_state = mean + stddev*torch.randn_like(mean)  # sample from normal distribution
        return {"det_state": det_state, "stoch_state": stoch_state, "mean": mean, "stddev": stddev}

    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | h_t, e_t)"""
        prior = self._prior(prev_state, prev_action)
        post_mean, post_stddev = self.posterior_model(prior["det_state"], emb_observation)
        post_state = post_mean + post_stddev*torch.randn_like(post_mean)  # sample from normal distribution
        return {"det_state": prior["det_state"], "stoch_state": post_state, "mean": post_mean, "stddev": post_stddev}

    def dec(self, state):
        """o_t ~ p(o_t | h_t, s_t)"""
        return self.decoder(state)

class RSSMPrior(nn.Module):
    def __init__(self, min_stddev, state_size, action_size, hidden_size=None):
        super().__init__()
        self.min_stddev = min_stddev
        self.input_size = state_size["stoch_state"] + action_size
        if hidden_size is None:  # size of hidden layer for stoch. state
            hidden_size = state_size["det_state"]
        # GRU hidden_size is size of hidden state (i.e. deterministic state)
        self.gru = nn.GRUCell(self.input_size, state_size["det_state"])        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size["det_state"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*state_size["stoch_state"])
        )
    
    def forward(self, state, action):
        input = torch.cat((state["stoch_state"], action), dim=1)
        det_state = self.gru(input, state["det_state"])
        output = self.linear_relu_stack(det_state)
        mean, stddev = output.chunk(2, dim=-1)
        stddev = nn.functional.softplus(stddev) + self.min_stddev
        return det_state, mean, stddev

class RSSMPosterior(nn.Module):
    def __init__(self, min_stddev, state_size, embedded_size, hidden_size=None):
        super().__init__()
        self.input_size = state_size["det_state"] + embedded_size
        if hidden_size is None:
            hidden_size = 2*state_size["stoch_state"]
        self.min_stddev = min_stddev
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*state_size["stoch_state"])
        )

    def forward(self, det_state, embedded):
        input = torch.cat((det_state, embedded), dim=1)
        output = self.linear_relu_stack(input)
        mean, stddev = output.chunk(2, dim=-1)
        stddev = nn.functional.softplus(stddev) + self.min_stddev
        return mean, stddev