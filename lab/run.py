from typing import Dict
from lab import Agent, Env, Trainer
from lab.utilities.cli import cli


@cli(config_path="lab/config/")
def main(cfg: Dict) -> None:
    print(cfg)
    trainer = Trainer.from_config(cfg)
    env_fn = Env.from_config(cfg)  # env_fn/agent_fn are factory methods to create new envs/agents
    init_env = env_fn()
    agent_fn = Agent.from_config(cfg, init_env.spaces)

    if cfg["test"]:
        trainer.test(agent_fn, env_fn)
    else:
        trainer.train(agent_fn, env_fn)


if __name__ == "__main__":
    main()
