from typing import NamedTuple
import jax
from jax import numpy as jnp
import numpy as np


class SkillRewardTracker(NamedTuple):
    best_skill: jnp.ndarray
    best_score: np.float32
    score_sum: np.float32
    score_step: int
    current_skill: jnp.ndarray
    search_steps: int
    change_interval: int
    low: float
    update: bool


def constant_fixed_skill(skill_dim: int,) -> jnp.ndarray:
    return jnp.ones((skill_dim,), dtype=jnp.float32) * 0.5


def random_search_skill(
    skill_dim: int,
    global_timestep: int,
    skill_tracker: SkillRewardTracker,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    if global_timestep >= skill_tracker.search_steps:
        return skill_tracker.best_skill
    return jax.random.uniform(key, shape=(skill_dim, ), minval=0., maxval=1.)


def random_grid_search_skill(key: jax.random.PRNGKey,
                             skill_dim: int,
                             global_timestep: int,
                             skill_tracker: SkillRewardTracker,
                             **kwargs) -> jnp.ndarray:
    if global_timestep >= skill_tracker.search_steps:
        return skill_tracker.best_skill
    increment = (1 - skill_tracker.low) / (skill_tracker.search_steps // skill_tracker.change_interval)
    start = global_timestep // skill_tracker.change_interval * increment
    end = (global_timestep // skill_tracker.change_interval + 1) * increment
    return jax.random.uniform(key, shape=(skill_dim,), minval=start, maxval=end)


def grid_search_skill(skill_dim: int, global_timestep: int, skill_tracker: SkillRewardTracker) -> jnp.ndarray:
    if global_timestep >= skill_tracker.search_steps:
        return skill_tracker.best_skill
    return jnp.ones((skill_dim,)) * jnp.linspace(
        -1.,
        0.,
        num=skill_tracker.search_steps // skill_tracker.change_interval
    )[global_timestep // skill_tracker.change_interval]

