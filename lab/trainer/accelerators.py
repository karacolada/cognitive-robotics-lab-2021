import functools
import time
import torch
from typing import Callable
import warnings


class TrainerAcceleratorsMixin:
    def _setup_accelerator(self, accelerator: str, device: str) -> None:
        if device == "cpu":
            if torch.cuda.is_available():
                warnings.warn(f"GPU is available, but device CPU was selected.")
                time.sleep(5)
            self.accelerator = CPUAccelerator()

        elif device == "cuda":
            self.accelerator = GPUAccelerator(accelerator)

        else:
            raise ValueError(f"Unknown device '{device}' given. Choice of ['cpu', 'cuda'].")


class CPUAccelerator:
    @staticmethod
    def setup(agent_fn: Callable) -> Callable:
        def move_to_cpu(agent_fn: Callable):
            agent = agent_fn()
            agent.to_device(torch.device("cpu"))
            return agent
        return functools.partial(move_to_cpu, agent_fn)


class GPUAccelerator:
    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator

    def setup(self, agent_fn: Callable) -> Callable:
        if not self.accelerator:
            def move_to_cuda0(agent_fn: Callable):
                agent = agent_fn()
                agent.to_device(torch.device("cuda:0"))
                return agent
            return functools.partial(move_to_cuda0, agent_fn)
        else:
            raise NotImplementedError
