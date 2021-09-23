import torch
from lab.models.dynamics_models.encoder import ImageEncoder

batch_size = 5
input = torch.randn(batch_size, 3, 64, 64)

embedded_size = 200
enc = ImageEncoder(embedded_size)

output = enc.forward(input)
print("Size of embedded batch: {}".format(output.size()))