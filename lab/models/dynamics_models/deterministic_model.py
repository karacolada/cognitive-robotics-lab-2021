from lab.models.dynamics_models.encoder import Encoder
from lab.models.dynamics_models.decoder import Decoder
from lab.models.dynamics_models.latent_dynamics_model import LatentDynamicsModel


class DeterministicModel(LatentDynamicsModel):
    def __init__(self, type, **kwargs):
        super().__init__()
        self.encoder = Encoder(type, **kwargs)
        self.decoder = Decoder(type, **kwargs)

    def _enc(self, observation):
        """e_t = enc(o_t)"""
        return self.encoder.forward(observation)

    def _prior(self, prev_state, prev_action):
        """s_t ~ p(s_t | s_t-1, a_t-1)"""
        # sample from learned transition model
        raise NotImplementedError

    def _posterior(self, prev_state, prev_action, emb_observation):
        """s_t ~ q(s_t | s_t-1, a_t-1, e_t)"""
        raise NotImplementedError

    def dec(self, state):
        """o_t ~ p(o_t | s_t)"""
        return self.decoder.forward(state)