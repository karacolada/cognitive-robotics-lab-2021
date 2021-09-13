import numpy as np
from typing import List
import torch
from lab.replay_buffer.ndarray_tuple import ndarray_tuple
from lab.replay_buffer.properties import ReplayBufferProperties


def _is_goal_conditioned(transition: ndarray_tuple) -> bool:
    return transition.goal.shape[-1] > 0


class ReplayBuffer(ReplayBufferProperties):
    def __init__(self, buffer_size: int, image_based: bool) -> None:
        self.buffer_size = buffer_size
        self.image_based = image_based
        self._storage = None
        self.full = False
        self.idx = 0

    def __len__(self) -> int:
        if self.full:
            return self.buffer_size
        else:
            return self.idx

    def store_transition(self, transition) -> None:
        """Store a single transition to the buffer."""
        if self._storage is None:
            self._init_from_sample_transition(transition)

        transition = self._process_transition_for_storage(transition)
        self._storage[self.idx] = transition
        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0

    def store_episode(self, episode) -> None:
        """Store a complete episode of transitions to the buffer."""
        raise NotImplementedError

    def sample(self, batch_size: int = 1, sequence_len: int = None) -> List:
        """Sample batch of transitions or batch of sequences if sequence_len is not None."""
        if sequence_len:
            idxs = np.asarray([self._sample_idx(sequence_len) for _ in range(batch_size)])
            transitions_batch = self._retrieve_batch(idxs, batch_size, sequence_len)
        else:
            idxs = np.random.randint(0, self.buffer_size if self.full else self.idx, size=batch_size)
            transitions_batch = self._storage[idxs]

        transitions_batch = self._to_tensors(transitions_batch)

        transitions_batch = self._process_transitions_for_learning(transitions_batch)
        return transitions_batch

    def to(self, device: torch.device) -> None:
        """Set device that sampled transitions will be on."""
        self.device = device

    def _init_from_sample_transition(self, transition) -> None:
        example_arrays = []
        transition_fields = []
        for field in transition._fields:
            value = getattr(transition, field)
            transition_fields.append(field)
            example_arrays.append(np.empty_like(np.array(value), shape=(self.buffer_size,) + np.array(value).shape))
            if example_arrays[-1].dtype == np.float64:
                example_arrays[-1] = example_arrays[-1].astype(np.float32)

        storage = ndarray_tuple("storage", transition_fields)
        self.transition_batch = ndarray_tuple("transition_batch", transition_fields)
        self._storage = storage(*example_arrays)

    def _process_transition_for_storage(self, transition):
        def postprocess_observation(observation):
            return np.clip(np.floor((observation + 0.5) * 255), 0, 255).astype(np.uint8)

        if self.image_based:
            transition = transition.replace("observation", postprocess_observation(transition.observation))
            if _is_goal_conditioned(transition):
                transition = transition.replace("goal", postprocess_observation(transition.goal))
        return transition

    def _process_transitions_for_learning(self, transitions):
        def preprocess_observation(observation):
            observation = (observation / 255.) - 0.5
            return observation

        if self.image_based:
            transitions = transitions.replace("observation", preprocess_observation(transitions.observation))
            if _is_goal_conditioned(transitions):
                transitions = transitions.replace("goal", preprocess_observation(transitions.goal))
        return transitions

    def _to_tensors(self, transitions):
        tensors = []
        for field in transitions:
            tensors.append(torch.from_numpy(field).to(device=self.device))
        tensor_transitions = self.transition_batch(*tensors)
        return tensor_transitions

    def _sample_idx(self, sequence_len: int) -> np.ndarray:
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_size if self.full else self.idx - sequence_len)
            idxs = np.arange(idx, idx + sequence_len) % self.buffer_size
            valid_idx = self.idx not in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    def _retrieve_batch(self, idxs: np.array, batch_size: int, sequence_len: int) -> ndarray_tuple:
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        transitions_batch = self._storage[vec_idxs]
        shapes = []
        for key, shape in transitions_batch.shape.items():
            if len(shape) > 1:
                shapes.append((sequence_len, batch_size) + shape[1:])
            else:
                shapes.append((sequence_len, batch_size,))  # Avoid adding an unnecessary dim for rewards and dones
        transitions_batch = transitions_batch.reshape(shapes)
        return transitions_batch
