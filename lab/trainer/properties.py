from abc import ABC
from typing import Dict
from lab.utilities.cli import _from_config


class TrainerProperties(ABC):

    @classmethod
    def from_config(cls, cfg: Dict):
        cls.hparams = cfg
        return _from_config(cls, cfg)
