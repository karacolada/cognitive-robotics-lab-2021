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
        det_state = self.prior_model(prev_state, prev_action)
        return {"det_state": det_state}

    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | h_t, e_t)"""
        prior = self._prior(prev_state, prev_action)
        posterior = self.posterior_model(prior["det_state"], emb_observation)
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
        # is it correct not to use an activation function on the GRU cell?
        # is it correct to have the feed-forward net behind it?
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size["det_state"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size["det_state"])
        )
    
    def forward(self, state, action):
        gru_out = self.gru(action, state["det_state"])
        det_state = self.linear_relu_stack(gru_out)
        return det_state

class DeterministicPosterior(nn.Module):
    def __init__(self, state_size, embedded_size, hidden_size=None):
        super().__init__()
        #self.input_size = state_size["det_state"] + embedded_size  # or action_size + embedded_size?
        self.input_size = embedded_size
        if hidden_size is None:
            hidden_size = state_size["det_state"]
        self.gru = nn.GRUCell(self.input_size, state_size["det_state"])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size["det_state"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size["det_state"])
        )

    def forward(self, det_state, embedded):
        gru_out = self.gru(embedded, det_state)
        det_state = self.linear_relu_stack(gru_out)
        return det_state