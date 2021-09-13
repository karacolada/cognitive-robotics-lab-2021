from lab.environment.env import Env
from lab.environment.wrapper import EnvWrapper
from typing import Tuple, Dict
import numpy as np
import cv2


class VisualEnv(EnvWrapper):
    def __init__(self, env: Env, render_size: Tuple):
        super().__init__(env)
        self._render_size = render_size
        self.initialize()

    def initialize(self):
        if self.env.is_gym_env or self.env.is_dm_control_env:
            pass
        elif self.env.is_multiworld_env:
            from multiworld.core.image_env import ImageEnv
            self.env = ImageEnv(self.env, imsize=max(self._render_size))
        else:
            raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple:
        state, reward, done, info = self.env.step(action)
        observation = self.render_observation(state)
        return observation, reward, done, info

    def reset(self):
        state = self.env.reset()
        observation = self.render_observation(state)
        return observation

    def render_observation(self, state: np.ndarray) -> np.ndarray:
        if self.env.is_gym_env or self.env.is_dm_control_env:
            image = self._render_gym()
        elif self.env.is_multiworld_env:
            image = self._render_multiworld(state)
        else:
            raise NotImplementedError
        image = self._process_image_observation(image)
        return image

    def _render_gym(self) -> np.ndarray:
        if self.env.is_goal_conditioned:
            raise NotImplementedError("Returning a visual observation for a goal-conditioned gym env is not "
                                      "implemented yet.")
        else:
            image = self.env.render(mode="rgb_array")
        return image

    def _render_multiworld(self, state: Dict, keys_to_keep: Tuple = ("observation", "desired_goal")) -> Dict:
        image = {}
        for key in state.keys():
            if key in keys_to_keep:
                image[key] = state[key].transpose(1, 2, 0)
        return image

    def _resize_image_observation(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self._render_size, interpolation=cv2.INTER_LINEAR)
        return resized_image

    @staticmethod
    def _normalize_image_observation(image: np.ndarray, min_value: float = -0.5, max_value: float = 0.5) -> np.ndarray:
        assert np.min(image) >= 0 and np.max(image) <= 255 and image.dtype == np.uint8, \
            f"Error in input image min={np.min(image)}, max={np.max(image)}, dtype={image.dtype}."
        normalized_image = ((image / 255.) + min_value) / (max_value - min_value)
        assert np.max(normalized_image) <= max_value and np.min(normalized_image) >= min_value, \
            f"Error normalizing the image min={np.min(normalized_image)}, max={np.max(normalized_image)}."
        return normalized_image

    @staticmethod
    def _adjust_image_observation(image: np.ndarray) -> np.ndarray:
        image = image.transpose(2, 0, 1)  # Put channel first as normal for torch
        image = image.astype(np.float32)  # Cast to regular precision of float32
        return image

    def _process_image_observation(self, image):
        if isinstance(image, np.ndarray):
            processed_image = self._resize_image_observation(image)
            processed_image = self._normalize_image_observation(processed_image)
            processed_image = self._adjust_image_observation(processed_image)
        elif isinstance(image, Dict):
            processed_image = {}
            for key in image.keys():
                processed_image[key] = self._resize_image_observation(image[key])
                processed_image[key] = self._normalize_image_observation(processed_image[key])
                processed_image[key] = self._adjust_image_observation(processed_image[key])
        else:
            raise ValueError("Expected image to be numpy array of Dict for goal-conditioned env, but found neither.")
        return processed_image

