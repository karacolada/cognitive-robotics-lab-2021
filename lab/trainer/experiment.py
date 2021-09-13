from typing import Callable
from lab.agent.agent import Agent
from lab.environment.env import Env
from lab.trainer.logger import Logger
import torch
import numpy as np
import random


class Experiment:
    def __init__(self, agent_fn: Callable, env_fn: Callable, logger: Logger, num_runs: int, resume_from_checkpoint: str,
                 seed: int, verbose: bool) -> None:
        self.agent_fn = agent_fn
        self.env_fn = env_fn
        self.logger = logger
        self.num_runs = num_runs
        self.resume_from_checkpoint = resume_from_checkpoint
        self.seeds = self._generate_seed_list(seed, self.num_runs)
        self.verbose = verbose
        self._experiment_phases = []
        self._started_runs = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._started_runs < self.num_runs:
            self._started_runs += 1
            current_env = self.env_fn()
            self._seed_everything(current_env, seed=self.seeds[self._started_runs - 1])
            current_agent = self.agent_fn()
            if self.resume_from_checkpoint:
                current_agent.load(self.resume_from_checkpoint)
            current_agent.connect_logger(self.logger, run_id=self._started_runs)
            self._print_summary(current_agent, current_env)
            return Run(self._process_fn, current_agent, current_env, self.logger, run_id=self._started_runs)
        else:
            raise StopIteration

    def add_phase(self, phase: Callable) -> None:
        self._experiment_phases.append(phase)

    def _process_fn(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> None:
        for phase in self._experiment_phases:
            yield phase(agent, env, logger, run_id)

    @staticmethod
    def _generate_seed_list(seed: int, num_seeds: int):
        random.seed(a=seed)
        seed_list = random.sample(range(0, 1000), num_seeds)
        return seed_list

    @staticmethod
    def _seed_everything(env: Env, seed: int):
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _print_summary(self, agent: Agent, env: Env) -> None:
        if self._started_runs == 1:
            print(f"Running {agent.name} on {env.env_name}:")
            agent.summarize()

class Run:
    def __init__(self, process_fn: Callable, agent: Agent, env: Env, logger: Logger, run_id: int) -> None:
        self._process_fn = process_fn
        self.agent = agent
        self.env = env
        self.logger = logger
        self.run_id = run_id
        self._running = False

    def __iter__(self):
        self._running = True
        for phase in self._process_fn(self.agent, self.env, self.logger, self.run_id):
            for value in phase:
                yield value
