from abc import ABC
from lab.utilities.cli import _from_config
from typing import Dict, List
import pyglet
import gym
import functools
import numpy as np


def wrap_env(env, cfg: Dict):
    from lab.environment.action_repeat import ActionRepeat
    from lab.environment.datatype import DataType
    from lab.environment.goal_conditioned_env import GoalConditionedEnv
    from lab.environment.normalize_actions import NormalizeActions
    from lab.environment.time_limit import TimeLimit
    from lab.environment.visual_env import VisualEnv

    env.image_based = cfg["image_based"]
    env._render_size = tuple(cfg["render_size"])
    env.action_repeat = cfg["action_repeat"]
    env._time_limit = cfg["time_limit"]
    env = NormalizeActions(env)
    env = ActionRepeat(env, cfg["action_repeat"])
    if cfg["image_based"]:
        env = VisualEnv(env, tuple(cfg["render_size"]))
    if cfg["time_limit"] > 0:
        env = TimeLimit(env, cfg["time_limit"], cfg["action_repeat"])
    env = GoalConditionedEnv(env)
    env = DataType(env)
    return env


class EnvProperties(ABC):
    @classmethod
    def from_config(cls, cfg: Dict):
        def env_factory_method(cls, cfg: Dict):
            env = _from_config(cls, cfg)
            return wrap_env(env, cfg)
        return functools.partial(env_factory_method, cls, cfg)

    @property
    def action_space(self):
        if self.is_gym_env or self.is_multiworld_env:
            return self.env.action_space
        elif self.is_dm_control_env:
            action_space = gym.spaces.Box(self.env.action_spec().minimum, self.env.action_spec().maximum,
                                          self.env.action_spec().shape)
            return action_space
        else:
            raise NotImplementedError

    @property
    def observation_space(self):
        if self.is_gym_env or self.is_multiworld_env:
            return self.env.observation_space
        elif self.is_dm_control_env:
            if len(self.env.observation_spec().keys()) == 1:
                observation_spec = list(self.env.observation_spec().values())[0]
                observation_space = gym.spaces.Box(observation_spec.minimum, observation_spec.maximum,
                                                   observation_spec.shape)
            else:
                ndim = sum([np.int(np.prod(self.env.observation_spec()[key].shape)) for key in
                           self.env.observation_spec()])
                observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(ndim,))
            return observation_space

    @property
    def action_size(self):
        if self.is_gym_env or self.is_multiworld_env or self.is_dm_control_env:
            assert len(self.action_space.shape) == 1, "Action space should be a one-dimensional vector."
            return self.action_space.shape[0]
        else:
            raise NotImplementedError

    @property
    def has_discrete_actions(self):
        if self.is_gym_env or self.is_multiworld_env:
            return isinstance(self.action_space, gym.spaces.Discrete)
        if self.is_dm_control_env:
            return False
        else:
            raise NotImplementedError

    @property
    def has_continuous_actions(self):
        if self.is_gym_env or self.is_multiworld_env:
            return isinstance(self.action_space, gym.spaces.Box)
        elif self.is_dm_control_env:
            return True
        else:
            raise NotImplementedError

    @property
    def is_goal_conditioned(self):
        if self.is_gym_env or self.is_multiworld_env or self.is_dm_control_env:
            return isinstance(self.observation_space, gym.spaces.dict.Dict)
        else:
            raise NotImplementedError

    @property
    def goal_size(self):
        if self.image_based:
            if self.is_goal_conditioned:
                return self.observation_size
        else:
            if self.is_goal_conditioned:
                assert len(self.observation_space["desired_goal"].shape) == 1, "Desired goal should be a " \
                                                                                    "one-dimensional vector."
                return self.observation_space["desired_goal"].shape[0]

    @property
    def observation_size(self):
        if self.is_gym_env or self.is_multiworld_env or self.is_dm_control_env:
            if self.image_based:
                return (3,) + self._render_size
            else:
                assert len(self.observation_space.shape) == 1, "Observation space should be a one-dimensional vector."
                if self.is_goal_conditioned:
                    return self.observation_space["observation"].shape[0]
                else:
                    return self.observation_space.shape[0]

    @property
    def max_num_actions(self):
        if self.is_gym_env or self.is_multiworld_env or self.is_dm_control_env:
            try:
                gym_max_steps = self.env._max_episode_steps
                return min(gym_max_steps, self._time_limit)
            except AttributeError:
                return self._time_limit
        else:
            raise NotImplementedError

    @staticmethod
    def _connect_display() -> None:
        try:
            pyglet.canvas.get_display()
        except pyglet.canvas.xlib.NoSuchDisplayException:
            from pyvirtualdisplay import Display
            display = Display(visible=False, size=(1400, 900))
            display.start()

    @property
    def is_gym_env(self):
        return self.env_type == "gym"

    @property
    def is_multiworld_env(self):
        return self.env_type == "multiworld"

    @property
    def is_dm_control_env(self):
        return self.env_type == "dm_control"

    @property
    def spaces(self):
        return_properties = ["observation_size", "action_size", "image_based", "has_continuous_actions",
                             "is_goal_conditioned", "goal_size"]
        spaces = {}
        for key in return_properties:
            spaces[key] = getattr(self, key)
        return spaces

    @property
    def multiworld_envs(self) -> List:
        mujoco_envs = ['SawyerPush-v0', 'SawyerDoorOpen-v0', 'SawyerPickup-v0', 'SawyerReachXYEnv-v1',
                       'SawyerReachXYZEnv-v0', 'SawyerReachTorqueEnv-v0', 'Image48SawyerReachXYEnv-v1',
                       'Image84SawyerReachXYEnv-v1', 'SawyerPushAndReachEnvEasy-v0', 'SawyerPushAndReachEnvMedium-v0',
                       'SawyerPushAndReachEnvHard-v0', 'SawyerPushAndReachArenaEnv-v0',
                       'SawyerPushAndReachArenaResetFreeEnv-v0', 'SawyerPushAndReachSmallArenaEnv-v0',
                       'SawyerPushAndReachSmallArenaResetFreeEnv-v0', 'SawyerPushNIPS-v0', 'SawyerPushNIPSHarder-v0',
                       'SawyerDoorHookResetFreeEnv-v0', 'SawyerPickupEnv-v0', 'SawyerPickupResetFreeEnv-v0',
                       'SawyerPickupEnvYZ-v0', 'SawyerPickupTallEnv-v0', 'SawyerPickupWideEnv-v0',
                       'SawyerPickupWideResetFreeEnv-v0', 'SawyerPickupTallWideEnv-v0', 'SawyerPickupEnvYZEasy-v0',
                       'SawyerPickupEnvYZEasyFewGoals-v0', 'SawyerPickupEnvYZEasyImage48-v0',
                       'SawyerDoorHookResetFreeEnvImage48-v1', 'SawyerPushNIPSEasy-v0', 'SawyerPushNIPSEasyImage48-v0',
                       'SawyerDoorHookResetFreeEnv-v1', 'SawyerReachXYZEnv-v2', 'LowGearAnt-v0', 'AntXY-v0',
                       'AntXY-NoContactSensors-v0', 'AntXY-LowGear-v0', 'AntXY-LowGear-NoContactSensors-v0',
                       'AntFullPositionGoal-v0']
        pygame_envs = ['Point2DLargeEnv-v1', 'Point2DEasyEnv-v1', 'Point2DLargeEnv-offscreen-v0',
                       'Point2DLargeEnv-onscreen-v0', 'Point2D-Box-Wall-v1', 'Point2D-Big-UWall-v1',
                       'Point2D-Easy-UWall-v1', 'Point2D-Easy-UWall-v2', 'Point2D-Easy-UWall-Hard-Init-v2',
                       'Point2D-FlatWall-v2', 'Point2D-FlatWall-Hard-Init-v2', 'Point2D-Easy-UWall-WhiteBackground-v1',
                       'Point2D-Easy-UWall-Hard-Init-WhiteBackground-v1', 'Point2D-FlatWall-WhiteBackground-v1',
                       'Point2D-FlatWall-Hard-Init-WhiteBackground-v1', 'Point2D-ImageFixedGoal-v0', 'Point2D-Image-v0',
                       'FiveObject-PickAndPlace-RandomInit-1D-v1', 'FourObject-PickAndPlace-RandomInit-1D-v1',
                       'ThreeObject-PickAndPlace-RandomInit-1D-v1', 'TwoObject-PickAndPlace-RandomInit-1D-v1',
                       'OneObject-PickAndPlace-RandomInit-1D-v1', 'FiveObject-PickAndPlace-OriginInit-1D-v1',
                       'FourObject-PickAndPlace-OriginInit-1D-v1', 'ThreeObject-PickAndPlace-OriginInit-1D-v1',
                       'TwoObject-PickAndPlace-OriginInit-1D-v1', 'OneObject-PickAndPlace-OriginInit-1D-v1',
                       'FiveObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                       'FourObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                       'ThreeObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                       'TwoObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                       'OneObject-PickAndPlace-OnRandomObjectInit-1D-v1', 'FiveObject-PickAndPlace-RandomInit-2D-v1',
                       'FourObject-PickAndPlace-RandomInit-2D-v1', 'ThreeObject-PickAndPlace-RandomInit-2D-v1',
                       'TwoObject-PickAndPlace-RandomInit-2D-v1', 'OneObject-PickAndPlace-RandomInit-2D-v1',
                       'FiveObject-PickAndPlace-OriginInit-2D-v1', 'FourObject-PickAndPlace-OriginInit-2D-v1',
                       'ThreeObject-PickAndPlace-OriginInit-2D-v1', 'TwoObject-PickAndPlace-OriginInit-2D-v1',
                       'OneObject-PickAndPlace-OriginInit-2D-v1', 'FiveObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                       'FourObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                       'ThreeObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                       'TwoObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                       'OneObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                       'FourObject-PickAndPlace-BigBall-RandomInit-2D-v1',
                       'ThreeObject-PickAndPlace-BigBall-RandomInit-2D-v1',
                       'TwoObject-PickAndPlace-BigBall-RandomInit-2D-v1',
                       'OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
                       'ZeroObject-PickAndPlace-BigBall-RandomInit-2D-v1',
                       'FourObject-PickAndPlace-SameSize-RandomInit-2D-v1',
                       'ThreeObject-PickAndPlace-SameSize-RandomInit-2D-v1',
                       'TwoObject-PickAndPlace-SameSize-RandomInit-2D-v1',
                       'OneObject-PickAndPlace-SameSize-RandomInit-2D-v1',
                       'ZeroObject-PickAndPlace-SameSize-RandomInit-2D-v1']
        multiworld_envs = mujoco_envs + pygame_envs
        return multiworld_envs

    @staticmethod
    def _obs_dmc2gym(spec_obs):
        # Adapted from: https://github.com/martinseilair/dm_control2gym
        if len(spec_obs.keys()) == 1:
            return list(spec_obs.values())[0]
        else:
            ndim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
            space_obs = np.zeros((ndim,))
            i = 0
            for key in spec_obs:
                space_obs[i:i + np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
                i += np.prod(spec_obs[key].shape)
            return space_obs
