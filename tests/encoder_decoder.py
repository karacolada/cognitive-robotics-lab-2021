import torch
from lab.models.dynamics_models.encoder import ImageEncoder
from lab.models.dynamics_models.decoder import ImageDecoder

input = torch.randn(3, 64, 64)
input = input.unsqueeze(0)

embedded_size = 200
enc = ImageEncoder(embedded_size)

embedded = enc.forward(input)
print("Size of embedded batch: {}".format(embedded.size()))

dec = ImageDecoder(embedded_size)

observation = dec.forward(embedded)
print("Size of observed batch: {}".format(observation.size()))