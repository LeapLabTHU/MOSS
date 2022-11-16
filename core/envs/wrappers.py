from typing import Optional, NamedTuple
from collections import deque

import dm_env
from dm_env import StepType, TimeStep
from jax import numpy as jnp
import numpy as np


class InformativeTimeStep(NamedTuple):
    step_type: StepType
    reward: float
    discount: float
    observation: jnp.ndarray
    action: jnp.ndarray
    mode: int
    current_timestep: int
    max_timestep: int

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


def timestep2informative_timestep(
    timestep: TimeStep,
    action: Optional[jnp.ndarray] = None,
    mode: Optional[int] = None,
    current_timestep: Optional[int] = None,
    max_timestep: Optional[int] = None,) -> InformativeTimeStep:
    return InformativeTimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward,
        discount=timestep.discount,
        observation=timestep.observation,
        action=action,
        mode=mode,
        current_timestep=current_timestep,
        max_timestep=max_timestep,
    )


class Wrapper(dm_env.Environment):
    def __init__(self, env: dm_env.Environment):
        self._env = env
        # inherent some attributes from env, like time counter, etc
        for attr, val in vars(self._env).items():
            if attr not in vars(self):
                setattr(self, attr, val)

    def action_spec(self):
        return self._env.action_spec()

    @property
    def timestep(self):
        return self._env._timestep

    @property
    def max_timestep(self):
        return self._env.max_timestep

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStacker(Wrapper):
    def __init__(self, env: dm_env.Environment, frame_stack: int = 3):
        super().__init__(env)
        self._observation = deque(maxlen=frame_stack)
        self.n_stacks = frame_stack

    def observation_spec(self):
        single_observation_spec = self._env.observation_spec()
        new_shape = list(single_observation_spec.shape)
        new_shape[self._env._channel_axis] = new_shape[self._env._channel_axis] * self.n_stacks
        return dm_env.specs.Array(
            shape=tuple(new_shape),
            dtype=single_observation_spec.dtype,
            name=single_observation_spec.name
        )

    def reset(self,) -> dm_env.TimeStep:
        timestep = self._env.reset()
        # stack n_stacks init frames for first observation
        for _ in range(self.n_stacks):
            self._observation.append(timestep.observation)
        return timestep._replace(
            observation=np.concatenate(self._observation, axis=self._env._channel_axis))

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._env.step(action)
        self._observation.append(timestep.observation)
        return timestep._replace(
            observation=np.concatenate(self._observation, axis=self._env._channel_axis))


class ActionRepeater(Wrapper):
    def __init__(self, env: dm_env.Environment, nrepeats: int = 3):
        super().__init__(env)
        self._nrepeats = nrepeats

    def reset(self,) -> dm_env.TimeStep:
        return self._env.reset()

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        for _ in range(self._nrepeats):
            timestep = self._env.step(action)
        return timestep


class InformativeTimestepWrapper(Wrapper):
    def __init__(self, env: dm_env.Environment):
        super().__init__(env)

    def reset(self,) -> InformativeTimeStep:
        timestep = self._env.reset()
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return timestep2informative_timestep(
            # this namedtuple contains obs, reward, etc.
            timestep,
            action=action,
            # this is the time spent in this episode
            current_timestep=self._env.timestep,
            max_timestep=self._env.max_timestep,
        )

    def step(self, action: np.ndarray) -> InformativeTimeStep:
        timestep = self._env.step(action)
        return timestep2informative_timestep(
            timestep,
            action=action,
            current_timestep=self._env.timestep,
            max_timestep=self._env.max_timestep,
        )


class RewardScaler(Wrapper):
    def __init__(self, env: dm_env.Environment, reward_scale: float):
        super().__init__(env)
        self._reward_scale = reward_scale

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._env.step(action)
        return dm_env.TimeStep(
            step_type=timestep.step_type, reward=timestep.reward * self._reward_scale,
            discount=timestep.discount, observation=timestep.observation
        )


class DMCTimeWrapper(Wrapper):
    def __init__(self, env: dm_env.Environment,):
        super().__init__(env)
        self._env = env
        self._timestep = 0
        self.action_shape = self._env.action_spec().shape

    @property
    def max_timestep(self,) -> int:
    # last step
        if hasattr(self._env, '_time_limit'):
            return self._env._time_limit / self._env._task.control_timestep
        if hasattr(self._env, '_step_limit'):
            return self._env._step_limit

    @property
    def timestep(self,) -> int:
    # current in the episode
        return self._timestep

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self._timestep += 1
        return self._env.step(action)

    def reset(self,) -> dm_env.TimeStep:
        self._timestep = 0
        return self._env.reset()
