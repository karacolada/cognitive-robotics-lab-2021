from abc import ABC
from lab.utilities.cli import _from_config
from typing import Dict


class ReplayBufferProperties(ABC):

    @classmethod
    def from_config(cls, cfg: Dict):
        return _from_config(cls, cfg)
