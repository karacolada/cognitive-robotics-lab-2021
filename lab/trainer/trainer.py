from lab.trainer.accelerators import TrainerAcceleratorsMixin
from lab.trainer.experiment import Experiment
from lab.trainer.logger import Logger
from lab.trainer.phases import SeedEpisodePhase, TrainingPhase, TestPhase
from lab.trainer.properties import TrainerProperties
from typing import Callable, Dict


class Trainer(TrainerProperties, TrainerAcceleratorsMixin):
    def __init__(self, accelerator: str, checkpoint_interval: int, device: str, disable_video_logging: bool,
                 log_dir: str, log_to_json: bool, log_to_tensorboard: bool, log_to_wandb: bool, num_runs: int,
                 resume_from_checkpoint: str, seed: int, seed_episodes: int, test_episodes: int, test_interval: int,
                 train_episodes: int, update_steps: int, verbose: bool) -> None:
        self.checkpoint_interval = checkpoint_interval
        self.disable_video_logging = disable_video_logging
        self.num_runs = num_runs
        self.resume_from_checkpoint = resume_from_checkpoint
        self.seed = seed
        self.seed_episodes = seed_episodes
        self.test_episodes = test_episodes
        self.test_interval = test_interval
        self.train_episodes = train_episodes
        self.update_steps = update_steps
        self.verbose = verbose
        self._setup_accelerator(accelerator, device)
        self.logger = Logger(log_dir, num_runs, log_to_json, log_to_tensorboard, log_to_wandb)

    def train(self, agent_fn: Callable, env_fn: Callable):
        """Trains the agent on a given environment for the specified number of episodes."""
        agent_fn = self.accelerator.setup(agent_fn)
        self._run_train(agent_fn, env_fn)

    def test(self, agent_fn: Callable, env_fn: Callable):
        """Performs specified number of test episodes and reports averaged results."""
        if not self.resume_from_checkpoint:
            print("You called test without loading a trained model.")
        agent_fn = self.accelerator.setup(agent_fn)
        self._run_test(agent_fn, env_fn)

    def _run_train(self, agent_fn: Callable, env_fn: Callable) -> Dict:
        experiment = self._create_experiment(agent_fn, env_fn)
        experiment.add_phase(SeedEpisodePhase(self.seed_episodes))
        experiment.add_phase(TrainingPhase(self.checkpoint_interval, self.disable_video_logging, self.num_runs,
                                           self.test_episodes, self.test_interval, self.train_episodes,
                                           self.update_steps))
        results = self._run_experiment(experiment)
        return results

    def _run_test(self, agent_fn: Callable, env_fn: Callable):
        experiment = self._create_experiment(agent_fn, env_fn)
        experiment.add_phase(TestPhase(self.disable_video_logging, self.test_episodes))
        results = self._run_experiment(experiment)
        return results

    def _run_experiment(self, experiment: Experiment) -> Dict:
        for i, run in enumerate(experiment):
            results = {}
            for score in run:
                for key in score.keys():
                    if key not in results.keys():
                        results[key] = []
                    results[key].append(score[key])
            self.logger.to_tensorboard.add_hparams(self.hparams, results, run_id=i + 1)
        self._plot_results()
        return results

    def _create_experiment(self, agent_fn: Callable, env_fn: Callable) -> Experiment:
        return Experiment(agent_fn, env_fn, self.logger, self.num_runs, self.resume_from_checkpoint, self.seed,
                          self.verbose)

    def _plot_results(self) -> None:
        self.logger.to_matplotlib(self.logger._metrics, name="test_rewards")
