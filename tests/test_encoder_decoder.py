import torch
from lab.models.dynamics_models.encoder import ImageEncoder, VectorEncoder
from lab.models.dynamics_models.decoder import ImageDecoder, VectorDecoder

print("Test image:")

input = torch.randn(3, 64, 64)
input = input.unsqueeze(0)

embedded_size = 200
state_size = embedded_size
enc = ImageEncoder(embedded_size)

embedded = enc(input)
print("Size of embedded batch: {}".format(embedded.size()))

dec = ImageDecoder(state_size)

observation = dec(embedded)
print("Size of observed batch: {}".format(observation.size()))

print("Test vector:")

observation_size = 500

input = torch.randn(observation_size)
input = input.unsqueeze(0)  # make batch

enc = VectorEncoder(observation_size, embedded_size, hidden_size=300)

embedded = enc(input)
print("Size of embedded batch: {}".format(embedded.size()))

state_size = {"stoch_state":30, "det_state":25}
#state_size = {"det_state":25}
#state_size = {"stoch_state":30}
stoch_state = torch.randn(30)
det_state = torch.randn(25)
state = {"stoch_state":stoch_state, "det_state": det_state}

dec = VectorDecoder(observation_size, state_size, hidden_size=300)

observation = dec(state)
print("Size of observed batch: {}".format(observation.size()))