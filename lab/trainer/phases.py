from abc import ABC, abstractmethod
import cv2
from lab.agent.agent import Agent
from lab.environment.env import Env
from lab.trainer.logger import Logger
import numpy as np
import time
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Dict, Generator, List, Tuple
from copy import deepcopy


class TrainerPhase(ABC):
    def __call__(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> Generator:
        return self._run(agent, env, logger, run_id)

    @abstractmethod
    def _run(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> Generator:
        """Returns a generator that iterates over scores such as train_rewards."""
        raise NotImplementedError


class SeedEpisodePhase(TrainerPhase):
    def __init__(self, seed_episodes: int) -> None:
        self.seed_episodes = seed_episodes

    def _run(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> Generator:
        for episode in range(1, self.seed_episodes + 1):
            interact(agent, env, logger, phase="seed", run_id=run_id)
            yield {"seed_episodes": episode}


class TestPhase(TrainerPhase):
    def __init__(self, disable_video_logging: bool, test_episodes: int) -> None:
        self.disable_video_logging = disable_video_logging
        self.test_episodes = test_episodes

    def _run(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> Generator:
        test_rewards = run_test_episodes(agent, env, logger, run_id, self.disable_video_logging, self.test_episodes,
                                         "Running test episodes")
        yield {"test_rewards": test_rewards}


class TrainingPhase(TrainerPhase):
    def __init__(self, checkpoint_interval: int, disable_video_logging: bool, num_runs: int, test_episodes: int,
                 test_interval: int, train_episodes: int, update_steps: int, delay: float = 0.001) -> None:
        self.checkpoint_interval = checkpoint_interval
        self.disable_video_logging = disable_video_logging
        self.num_runs = num_runs
        self.test_episodes = test_episodes
        self.test_interval = test_interval
        self.train_episodes = train_episodes
        self.update_steps = update_steps
        self.delay = delay

    def _run(self, agent: Agent, env: Env, logger: Logger, run_id: int) -> Generator:
        if logger.log_to_wandb:
            logger.to_wandb.watch(agent, run_id)
        episode_prog_bar = _init_episode_prog_bar(self.train_episodes, logger, run_id)
        train_rewards, test_rewards = [], []
        for episode in episode_prog_bar:
            time.sleep(self.delay)
            _decorate_episode_prog_bar(episode_prog_bar, logger, train_rewards, test_rewards, run_id, self.num_runs)
            training_prog_bar = tqdm(range(self.update_steps), desc="  └── Updating agent", leave=False)
            # Train / update the agent based on previous experience.
            avg_losses = {}
            for update_step in training_prog_bar:
                agent.time_to_log = update_step == 0  # Only allow logging on first update step
                batch = agent.sample()
                losses = agent.learn_on_batch(batch)
                training_prog_bar.set_postfix(losses)
                for key in losses.keys():
                    if key not in avg_losses.keys():
                        avg_losses[key] = []
                    avg_losses[key].append(losses[key])
            for key, value in avg_losses.items():
                logger.add(key, np.array(value).mean(), logger.train_episodes(run_id), prefix="losses", run_id=run_id)

            training_prog_bar.close()

            # Interact with the environment for one episode to collect experience.
            logs, _, _ = interact(agent, env, logger, phase="train", run_id=run_id,
                            tqdm_desc="  └── Running training episode", log=("reward",))
            logger.add("rewards", logs["reward"], logger.train_episodes(run_id), prefix="train", run_id=run_id)
            train_rewards.append(logs["reward"])
            yield {"train_rewards": train_rewards[-1]}

            # Run test episodes and log results.
            if episode % self.test_interval == 0:
                test_rewards.append(run_test_episodes(agent, env, logger, run_id, self.disable_video_logging,
                                                      self.test_episodes))
                yield {"test_rewards": test_rewards[-1]}

            # Save current model state.
            if episode % self.checkpoint_interval == 0:
                save_agent(agent, logger, run_id)
        episode_prog_bar.close()


def interact(agent: Agent, env: Env, logger: Logger, phase: str, run_id: int, tqdm_desc: str = None,
             num_interaction_steps: int = None, log: Tuple = ()):
    if num_interaction_steps is None:
        num_interaction_steps = env.max_num_actions // env.action_repeat
    if tqdm_desc:
        interaction_prog_bar = tqdm(range(num_interaction_steps), desc=tqdm_desc, leave=False)
    else:
        interaction_prog_bar = range(num_interaction_steps)

    logs = _init_logs(log)
    observation, goal = env.reset()
    for t in interaction_prog_bar:
        action = _select_action(agent, env, observation, goal, t, phase)
        next_observation, next_goal, reward, done, info = env.step(action)
        if phase != "test":
            agent.store_transition(observation, goal, action, reward, done, next_observation, next_goal)
        observation = next_observation
        goal = next_goal
        _update_logs(logs, agent, env, reward, observation, goal, t)
        if done:
            if tqdm_desc:
                interaction_prog_bar.close()
            break

    if phase != "test":
        logger.increment_train_steps(t + 1, run_id)
        logger.increment_train_episodes(run_id)

    for key in logs.keys():
        if isinstance(logs[key], List):
            logs[key] = np.stack(logs[key])
    return logs, observation, goal


def _select_action(agent: Agent, env: Env, observation: np.ndarray, goal: np.ndarray, t: int, phase) -> np.ndarray:
    if phase == "seed":
        action = env.sample_random_action()
    elif phase == "train":
        action = agent.get_no_grad_action(observation, goal, explore=True, episode_start=t == 0)
    elif phase == "test":
        action = agent.get_no_grad_action(observation, goal, explore=False, episode_start=t == 0)
    else:
        assert False, f"Expected phase to be in ['seed', 'train', 'test'], but got {phase}."
    return action


@torch.no_grad()
def run_test_episodes(agent: Agent, env: Env, logger: Logger, run_id: int, disable_video_logging: bool,
                      test_episodes: int, tqdm_desc: str = "  └── Running test episodes") -> np.ndarray:
    test_prog_bar = tqdm(range(test_episodes), desc=tqdm_desc, leave=False)
    logs = {}
    for test_episode in test_prog_bar:
        log = ["reward"]
        if not disable_video_logging:
            log.append("video")
            #if agent.model_based:
            #    log.append("reconstruction")
        current_logs, _, _ = interact(agent, env, logger, phase="test", run_id=run_id, log=tuple(log))
        for key in current_logs.keys():
            if key not in logs.keys():
                logs[key] = []
            logs[key].append(current_logs[key])

    if not disable_video_logging:
        _log_videos(agent, env, logger, logs, run_id)

    mean_test_reward = np.mean(np.array(logs["reward"]))
    logger.add("rewards", mean_test_reward, logger.train_episodes(run_id), prefix="test", run_id=run_id)
    return mean_test_reward


def save_agent(agent: Agent, logger: Logger, run_id: int) -> None:
    (logger.run_dirs[run_id - 1] / "saved_agents").mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), logger.run_dirs[run_id - 1] / "saved_agents" /
               f"algo={agent.name}_train_episode={str(logger.train_episodes(run_id))}.pt")


############ Logging methods ############

def _init_logs(log: Tuple) -> Dict:
    logs = {}
    if "video" in log:
        logs["video"] = []
    if "reconstruction" in log:
        logs["reconstruction"] = []
    if "reward" in log:
        logs["reward"] = 0
    return logs


def _update_logs(logs: Dict, agent: Agent, env: Env, reward: float, observation, goal, t: int):
    if "video" in logs.keys():
        _add_video(logs, env)
    if "reconstruction" in logs.keys():
        _add_reconstruction(logs, agent)
    if "reward" in logs.keys():
        _add_reward(logs, reward)


def _add_video(logs: Dict, env: Env, video_dims: Tuple = (256, 256)) -> None:
    logs["video"].append(env.render(mode="rgb_array", dims=video_dims) / 255.)


def _add_reconstruction(logs: Dict, agent: Agent, video_dims: Tuple = (256, 256)) -> None:
    reconstruction = np.clip(agent.dynamics_model.dec(agent.state["state"]).cpu().numpy()[0] + 0.5, 0., 1.)
    reconstruction = cv2.resize(reconstruction.transpose(1, 2, 0), video_dims).transpose(2, 0, 1)
    logs["reconstruction"].append(reconstruction)


def _add_reward(logs: Dict, reward: float) -> None:
    logs["reward"] += reward


def _log_videos(agent: Agent, env: Env, logger: Logger, logs: Dict, run_id: int) -> None:
    episode_videos = _make_video_grid(logs, logger)
    logger.add("episode_videos", episode_videos, logger.train_episodes(run_id), prefix="test", run_id=run_id)
    if agent.model_based:
        open_loop_video, open_loop_figure = _make_open_loop_prediction_visualizations(agent, env, logger, run_id)
        logger.add("open_loop_videos", open_loop_video, logger.train_episodes(run_id), prefix="test", run_id=run_id)
        logger.add("open_loop_rollouts", open_loop_figure, logger.train_episodes(run_id), prefix="test", run_id=run_id)


def _make_open_loop_prediction_visualizations(agent: Agent, env: Env, logger: Logger, run_id: int, context_len: int = 5,
                                              prediction_len: int = 45, video_dims: Tuple = (256, 256)):
    open_loop_predictions = _sample_open_loop_predictions(agent, env, logger, context_len, prediction_len, video_dims)
    open_loop_video = _make_open_loop_video(open_loop_predictions)
    open_loop_figure = logger.to_matplotlib._prediction_rollouts_fig(open_loop_predictions, context_len, run_id,
                                                                     logger.train_episodes(run_id))
    return open_loop_video, open_loop_figure


def _sample_open_loop_predictions(agent: Agent, env: Env, logger: Logger, context_len: int, prediction_len: int,
                                  video_dims: Tuple) -> Dict:
    logs, observation, goal = interact(agent, env, logger, "test", 0, num_interaction_steps=context_len,
                                       log=("video", "reconstruction"))
    state = deepcopy(agent.state)
    open_loop_predictions = {"true": [logs["video"]], "model": [logs["reconstruction"]]}

    for t in range(prediction_len):
        #print("state == agent.state:", all(state == agent.state))
        #print("state['state']['det_state'][0, 0:5]:", state['state']['det_state'][0, 0:5])
        #print("agent.state['state']['det_state'][0, 0:5]:", agent.state['state']['det_state'][0, 0:5])
        action = _select_action(agent, env, observation, goal, t + context_len, "test")
        next_observation, next_goal, reward, done, info = env.step(action)
        ground_truth = (env.render(mode="rgb_array", dims=video_dims) / 255.).numpy()
        open_loop_predictions["true"].append(np.expand_dims(ground_truth, axis=0))
        next_state = agent.dynamics_model(state, torch.from_numpy(action).unsqueeze(0).type(torch.float32).to(agent.device))
        prediction = agent.dynamics_model.dec(next_state["state"]).cpu().numpy()[0]  # Should be -0.5 to 0.5 according to visual_env
        prediction = np.clip((prediction + 0.5), 0., 1.)
        prediction = cv2.resize(prediction.transpose(1, 2, 0), video_dims).transpose(2, 0, 1)
        open_loop_predictions["model"].append(np.expand_dims(prediction, axis=0))
        observation, goal = next_observation, next_goal
        state = next_state
    for key, value in open_loop_predictions.items():
        open_loop_predictions[key] = np.concatenate(value, axis=0)

    return open_loop_predictions


def _make_open_loop_video(open_loop_predictions: Dict) -> np.ndarray:
    open_loop_video = np.concatenate([open_loop_predictions["true"], open_loop_predictions["model"]], axis=3)
    return open_loop_video







def _make_video_grid(logs: Dict, logger: Logger) -> torch.Tensor:
    assert "video" in logs.keys(), "Key 'video' missing from logs."
    num_videos = len(logs["video"])
    num_rows = max(int(np.floor(num_videos / 2)), 1)
    video = np.stack(logs["video"][0:2 * num_rows]) if num_videos > 1 else np.expand_dims(logs["video"][0], axis=0)

    if "reconstruction" in logs.keys():
        reconstruction = np.stack(logs["reconstruction"][0:2 * num_rows])
        video = np.concatenate([video, logger.to_tensorboard.normalize(torch.from_numpy(reconstruction)).numpy()],
                               axis=4)
    if "video_goal" in logs.keys():
        video_goal = np.stack(logs["video_goal"][0:2 * num_rows])
        video = np.concatenate([video, logger.to_tensorboard.normalize(torch.from_numpy(video_goal)).numpy()], axis=4)
    video_tensor = torch.from_numpy(video)
    video_grid = []
    for t in range(video.shape[1]):
        video_grid.append(make_grid(video_tensor[:, t], nrow=num_rows))
    video_grid = torch.stack(video_grid)
    return video_grid


############ TQDM helper methods ############

def _init_episode_prog_bar(train_episodes: int, logger: Logger, run_id: int) -> tqdm:
    episode_prog_bar = tqdm(range(logger.train_episodes(run_id) + 1, train_episodes + 1), total=train_episodes,
                            initial=logger.train_episodes(run_id) + 1, leave=True)
    return episode_prog_bar


def _decorate_episode_prog_bar(prog_bar: tqdm, logger: Logger, train_rewards: List, test_rewards: List, run_id: int,
                               num_runs: int, num_avg_steps: int = 3) -> None:
    if num_runs == 1:
        prog_bar.set_description(f"Episode {logger.train_episodes(run_id) + 1}")
    else:
        prog_bar.set_description(f"[Run {run_id}/{num_runs}] Episode {logger.train_episodes(run_id) + 1}")
    tqdm_dict = {}
    num_avg_train_episodes = min(len(train_rewards), num_avg_steps)
    num_avg_test_episodes = min(len(test_rewards), num_avg_steps)

    if num_avg_train_episodes > 0:
        avg_train_reward = np.array(train_rewards[-num_avg_train_episodes:]).mean()
        tqdm_dict["avg_train_reward"] = avg_train_reward
    if num_avg_test_episodes > 0:
        avg_test_reward = np.array(test_rewards[-num_avg_test_episodes:]).mean()
        tqdm_dict["avg_test_reward"] = avg_test_reward
    prog_bar.set_postfix(tqdm_dict)
