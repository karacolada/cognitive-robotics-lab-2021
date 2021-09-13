from abc import ABC
import torch
from torch import optim
from typing import List
from torch.nn import functional as F
import numpy as np


class AgentOptimizersMixin(ABC):
    def configure_criterion(self, criterion: str) -> None:
        self.criterion = getattr(F, criterion)

    def _configure_optimizers(self, optimizers: List, learning_rates: List, parameters: List,
                             lr_schedulers: List = None) -> None:
        assert len(optimizers) == len(learning_rates) == len(parameters), "Length of inputs does not match."

        self.optimizers = []
        for idx, opt in enumerate(optimizers):
            optimizer = getattr(optim, opt)
            self.optimizers.append(optimizer(parameters[idx], lr=learning_rates[idx]))
        if len(self.optimizers) == 1:
            self.optimizer = self.optimizers[0]

        if lr_schedulers and lr_schedulers[0]:
            self.lr_schedulers = []
            for idx, lrs in enumerate(lr_schedulers):
                lr_scheduler = getattr(optim.lr_scheduler, lrs)
                self.lr_schedulers.append(lr_scheduler(self.optimizers[idx]))
            if len(self.lr_schedulers) == 1:
                self.lr_scheduler = self.lr_schedulers[0]

    @staticmethod
    def polyak_update(target, main, rho):
        with torch.no_grad():
            for p_target, p in zip(target.parameters(), main.parameters()):
                p_target.data.mul_(rho)
                p_target.data.add_((1 - rho) * p.data)

    @staticmethod
    def seed_everything(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
