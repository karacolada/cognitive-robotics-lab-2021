import toml
from typing import Dict, List, Tuple, Callable
from pathlib import Path
import sys
import itertools
import functools
from copy import deepcopy
import inspect
import json
import os
from typing import Iterable


def cli(config_path: str) -> Callable:
    """Decorator for main function that hides parsing and starts experiments according to args."""
    def cli_decorator(main_fn: Callable) -> Callable:
        @functools.wraps(main_fn)
        def wrapper():
            base_config, config_groups = build_config_from_toml(config_path)
            configs = parse_args(base_config, config_groups)
            for cfg in configs:
                _pre_main_routine(cfg)
                main_fn(cfg)
                _post_main_routine()
        return wrapper
    return cli_decorator


def _pre_main_routine(cfg: Dict) -> None:
    experiment_dir = _generate_experiment_dir(cfg["log_dir"], cfg["algo"]["type"])
    cfg["log_dir"] = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    _save_config_to_json(experiment_dir, cfg)


def _post_main_routine() -> None:
    pass


def _generate_experiment_dir(log_dir: str, algo: str) -> str:
    base_dir = log_dir + "/" + algo
    i = 1
    while os.path.exists(base_dir + f"/experiment_{i}"):
        i += 1
    experiment_dir = base_dir + f"/experiment_{i}"
    return experiment_dir


def _save_config_to_json(experiment_dir: str, config: Dict) -> None:
    with open(experiment_dir + "/config.json", "w") as f:
        json.dump(config, f, indent=4)


def build_config_from_toml(config_path: str) -> Tuple:
    """Builds the initial config dict from toml files."""
    config = {}
    config_groups = []

    for base_config in Path(config_path).glob("*.toml"):
        config = {**config, **toml.load(base_config)}
    for config_group in [str(p).replace(config_path, "") for p in Path(config_path).glob("*") if p.is_dir()]:
        config_groups.append(config_group)
        config[config_group] = {}
        for subconfig_path in Path(config_path + config_group).glob("*toml"):
            subconfig = str(subconfig_path).replace(config_path + config_group + "/", "")[:-5]
            config[config_group][subconfig] = toml.load(subconfig_path)
    assert len(config) > 0, "No config information were added from TOML files"
    config = _process_datatypes(config)
    return config, config_groups


def parse_args(base_config: Dict, config_groups: List) -> List:
    """Combines initial config dict with commandline options and creates permutations for multiruns."""
    cl_args = _get_cl_args()
    group_configs = _get_group_configs(base_config.copy(), config_groups, cl_args)
    multirun_configs = _get_multirun_configs(group_configs, cl_args)
    return multirun_configs


def _process_datatypes(d: Dict, lists_to_tuples: bool = True) -> Dict:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _process_datatypes(v)
        else:
            if lists_to_tuples:
                if isinstance(v, list):
                    d[k] = tuple(v)
    return d


def _get_cl_args() -> Dict:
    args = sys.argv[1:]
    args_dict = {}
    for arg in args:
        key, value = arg.split("=", maxsplit=1)
        values = _process_command_line_values(value)
        args_dict[key] = values
    return args_dict


def _process_command_line_values(value: str) -> List:
    """Converts string to the respective datatypes."""
    values = value.split(",")
    constructors = [float, int]
    for idx, v in enumerate(values):
        if v == "True" or v == "true":
            values[idx] = True
        elif v == "False" or v == "false":
            values[idx] = False
        else:
            for c in constructors:
                try:
                    values[idx] = c(v)
                except ValueError:
                    pass
    return values


def _get_group_configs(base_config: Dict, config_groups: List, cl_args: Dict) -> List:
    config = base_config.copy()
    group_combinations = {}
    for group in config_groups:
        assert group in cl_args.keys(), f"No value for the config group '{group}' was given."
        group_combinations[group] = cl_args[group]
        config.pop(group, None)
        cl_args.pop(group, None)
    group_configs = []
    for combination in (dict(zip(group_combinations, v)) for v in itertools.product(*group_combinations.values())):
        curr_config = deepcopy(config)
        for key, value in combination.items():
            try:
                curr_config[key] = base_config[key][value]
            except KeyError:
                raise KeyError(f"The given value '{value}' is unavailable for the config group '{key}'")
            curr_config[key]["type"] = value
        group_configs.append(curr_config)
    return group_configs


def _get_multirun_configs(group_configs: List, cl_args: Dict) -> List:
    multirun_configs = []

    for config in group_configs:
        for cl_arg in (dict(zip(cl_args, v)) for v in itertools.product(*cl_args.values())):
            curr_config = deepcopy(config)
            for key, value in cl_arg.items():
                keys = key.split(".")
                if len(keys) == 1:
                    curr_config[keys[0]] = value
                elif len(keys) == 2:
                    curr_config[keys[0]][keys[1]] = value
                elif len(keys) == 3:
                    curr_config[keys[0]][keys[1]][keys[2]] = value
                else:
                    raise NotImplementedError
            multirun_configs.append(curr_config)
    return multirun_configs


def _from_config(cls, cfg: Dict):
    valid_kwargs = list(inspect.signature(cls.__init__).parameters)
    valid_kwargs.pop(0)
    missing_args = list(set(list(valid_kwargs)) - set(cfg.keys()))
    if len(missing_args) == 1:
        assert len(missing_args) == 0, f"{cls.__name__}.from_config() is missing the following required argument " \
                                       f"for __init__: '{missing_args[0]}'."
    elif len(missing_args) > 1:
        assert len(missing_args) == 0, f"{cls.__name__}.from_config() is missing the following required arguments " \
                                       f"for __init__: {missing_args}."
    valid_kwargs = {name: cfg[name] for name in valid_kwargs if name in cfg}
    return cls(**valid_kwargs)
