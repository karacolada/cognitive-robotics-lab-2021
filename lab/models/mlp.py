from torch import Tensor
import torch.nn as nn
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, layer_sizes: Tuple, hidden_activation: str = "relu", output_activation: str = "relu") -> None:
        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                                            for i in range(len(layer_sizes) - 1)])
        self.hidden_activation = getattr(nn.functional, hidden_activation)
        self.output_activation = None if output_activation == "identity" else getattr(nn.functional, output_activation)

    def forward(self, x: Tensor) -> Tensor:
        hidden = x
        for i, layer in enumerate(self.linear_layers):
            if i < len(self.linear_layers) - 1:
                hidden = self.hidden_activation(layer(x))
            else:
                output = layer(hidden)
                output = self.output_activation(output) if self.output_activation else output
        return output

