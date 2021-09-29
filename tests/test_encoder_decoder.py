import torch
from lab.models.dynamics_models.encoder import ImageEncoder, VectorEncoder
from lab.models.dynamics_models.decoder import ImageDecoder, VectorDecoder

state_size_rssm = {"stoch_state":30, "det_state":25}
state_size_rnn = {"det_state":25}
state_size_ssm = {"stoch_state":30}
stoch_state = torch.randn(30).unsqueeze(0)
det_state = torch.randn(25).unsqueeze(0)
state = {"stoch_state":stoch_state, "det_state": det_state}

embedded_size = 200

print("Encoder/Decoder test: image")

input = torch.randn(3, 64, 64)
input = input.unsqueeze(0)

enc = ImageEncoder(embedded_size)

embedded = enc(input)
print("Size of embedded batch: {}".format(embedded.size()))

print("----------------------\n    Decoder test: RSSM")
dec = ImageDecoder(state_size_rssm)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))

print("----------------------\n    Decoder test: RNN")
dec = ImageDecoder(state_size_rnn)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))

print("----------------------\n    Decoder test: SSM")
dec = ImageDecoder(state_size_ssm)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))

print("----------------------------\nEncoder/Decoder test: vector")

observation_size = 500

input = torch.randn(observation_size)
input = input.unsqueeze(0)  # make batch

enc = VectorEncoder(observation_size, embedded_size, hidden_size=300)

embedded = enc(input)
print("Size of embedded batch: {}".format(embedded.size()))

print("----------------------\n    Decoder test: RSSM")
dec = VectorDecoder(observation_size, state_size_rssm, hidden_size=300)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))

print("----------------------\n    Decoder test: RNN")
dec = VectorDecoder(observation_size, state_size_rnn, hidden_size=300)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))

print("----------------------\n    Decoder test: SSM")
dec = VectorDecoder(observation_size, state_size_ssm, hidden_size=300)
observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))