import functools
from abc import ABC
from lab.utilities.cli import _from_config
from typing import Dict
import os
import importlib
from pkgutil import iter_modules
from lab.replay_buffer.replay_buffer import ReplayBuffer


def get_agent(cfg: Dict):
    assert "algo" in cfg, "Required key 'algo' not in args."
    from lab import Agent
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "algos")
    for (module_loader, name, ispkg) in iter_modules(path=[pkg_dir]):
        importlib.import_module('..algos.' + name, __package__)
    agents = {cls.__name__.lower(): cls for cls in Agent.__subclasses__()}
    try:
        agent = agents[cfg["algo"]["type"]]
        agent.name = agent.__name__
        return agent
    except KeyError:
        raise ValueError(f"Unknown algorithm '{cfg['algo']['type']}' is not in possible algorithms {list(agents.keys())}.")


class AgentProperties(ABC):

    @classmethod
    def from_config(cls, cfg: Dict, env_spaces: Dict):
        def agent_factory_method(agt, cfg: Dict, env_spaces: Dict):
            agent = _from_config(agt, cfg["algo"])
            agent.initialize(env_spaces)
            agent.env_spaces = env_spaces
            agent.configure_optimizers()
            replay_buffer = ReplayBuffer.from_config(cfg)
            agent.connect_replay_buffer(replay_buffer)
            return agent
        agt = get_agent(cfg)
        return functools.partial(agent_factory_method, agt, cfg, env_spaces)

    @staticmethod
    def save__init__args(local_vars: Dict) -> None:
        self = local_vars["self"]
        if len(local_vars['args']) > 0:
            for key, value in local_vars["args"][0].items():
                if key != "self":
                    setattr(self, key, value)

    def _add_weights_dict(self, input_dict: Dict) -> Dict:
        for name, parameter in self.named_parameters():
            input_dict[name] = parameter.detach().cpu().numpy()

    def summarize(self, h_space: int = 6) -> None:
        def format_param_number(params: str) -> str:
            if len(params) > 6:
                params = params[:-6] + "." + params[-6] + " M"
            elif len(params) > 3:
                params = params[:-3] + " K"
            return params

        param_counts = {}
        max_len_name = 0
        max_len_params = 0
        for name, model in self.named_children():
            params = str(sum(p.numel() for p in model.parameters()))
            params = format_param_number(params)
            param_counts[name] = params
            max_len_name = max(max_len_name, len(name))
            max_len_params = max(max_len_params, len(params))

        total_params = format_param_number(str(sum(p.numel() for p in self.parameters())))
        total_trainable_params = format_param_number(str(sum(p.numel() for p in self.parameters() if p.requires_grad)))

        model_summaries = []
        for name, num_params in param_counts.items():
            model_summaries.append(name + (max_len_name - len(name) + h_space) * " " + str(num_params))

        summary = "_" * (max_len_name + 2 * h_space + max_len_params)
        summary += "\n" + self.name + "-Models" + " " * (max_len_name - len(self.name + "-Models") + h_space) + \
                   "Param #"
        summary += "\n" + "=" * (max_len_name + 2 * h_space + max_len_params)
        for i, line in enumerate(model_summaries):
            summary += "\n" + line
        summary += "\n" + "=" * (max_len_name + 2 * h_space + max_len_params)
        summary += "\n" + total_trainable_params + " " * (10 - len(total_trainable_params)) + "Trainable params"
        summary += "\n" + total_params + " " * (10 - len(total_params)) + "Total params"
        print(summary)
