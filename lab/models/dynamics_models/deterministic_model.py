import torch
from torch import nn
from lab.models.dynamics_models.encoder import Encoder
from lab.models.dynamics_models.decoder import Decoder
from lab.models.dynamics_models.latent_dynamics_model import LatentDynamicsModel
from lab.models.dynamics_models.recurrent_model import RSSMPrior, RSSMPosterior


class DeterministicModel(LatentDynamicsModel):
    def __init__(self, type, state_size, action_size, observation_size, embedded_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(type, observation_size, embedded_size)
        self.decoder = Decoder(type, observation_size, state_size, hidden_size)
        self.prior_model = DeterministicPrior(state_size, action_size, hidden_size)
        self.posterior_model = DeterministicPosterior(state_size, embedded_size, hidden_size)

    def _enc(self, observation):
        """e_t = enc(o_t)"""
        return self.encoder(observation)

    def _prior(self, prev_state, prev_action):
        """ h_t = f(h_t−1 , s_t−1 , a_t−1), s_t ~ p(s_t | h_t)"""
        self.hidden_prior, det_state = self.prior_model(prev_state, prev_action)
        return {"det_state": det_state}

    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | h_t, e_t)"""
        _ = self._prior(prev_state, prev_action)
        posterior = self.posterior_model(self.hidden_prior, emb_observation)
        return {"det_state": posterior}

    def dec(self, state):
        """o_t ~ p(o_t | h_t, s_t)"""
        return self.decoder(state)

class DeterministicPrior(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = state_size["det_state"]
        self.gru = nn.GRUCell(action_size, state_size["det_state"])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size["det_state"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size["det_state"])
        )
    
    def forward(self, state, action):
        gru_out = self.gru(action, state["det_state"])
        det_state = self.linear_relu_stack(gru_out)
        return gru_out, det_state

class DeterministicPosterior(nn.Module):
    def __init__(self, state_size, embedded_size, hidden_size=None):
        super().__init__()
        self.input_size = state_size["det_state"] + embedded_size
        if hidden_size is None:
            hidden_size = state_size["det_state"]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size["det_state"])
        )

    def forward(self, hidden_state, embedded):
        input = torch.cat((hidden_state, embedded), dim=1)
        det_state = self.linear_relu_stack(input)
        return det_state