from typing import Tuple, Any
from collections import OrderedDict
from functools import partial

import dm_env
import jax
import jax.numpy as jnp
import numpy as np

from core import agents
from core.envs import wrappers
from core import calculations


def episode_partition_mode_selector(
    time_step: wrappers.InformativeTimeStep,
    partitions: int = 2,
) -> bool:
    """
    True for control
    False for explore
    """
    max_time_step = time_step.max_timestep
    interval = max_time_step // partitions
    current_timestep = time_step.current_timestep
    return bool(current_timestep // interval % 2)


def get_meta_specs(skill_dim: int,
                   reward_free: bool
                   ) -> Tuple:
    # noinspection PyRedundantParentheses
    if reward_free:
        return (
            dm_env.specs.Array((skill_dim,), np.float32, 'skill'),
            dm_env.specs.Array((), np.bool, 'mode') # for pytorch replay buffer
        )
    else:
        return (
            dm_env.specs.Array((skill_dim,), np.float32, 'skill'),
        )

def init_meta(key,
              time_step: wrappers.InformativeTimeStep,
              reward_free: bool,
              skill_dim: int,
              partitions: int,
              search_mode = 'random_grid_search',
              skill_mode = 'half',
              skill_tracker: calculations.skill_utils.SkillRewardTracker=None,
              step: int = None,
) -> Tuple[OrderedDict, Any]:
    """
    :param key: only parameter needed in forward pass
    :param reward_free: defined as a constant during init of ddpg skill
    :param step: global step, at a certain step it only outputs the best skill
    :param skill_dim: defined as a constant during init of ddpg skill
    :param time_step: used to get current step in the episode for mode only
    :param skill_tracker: keep track in a NamedTuple of the best skill
    :return: during pretrain runing meta with skill and mode. Finetune return best skill in skill_tracker
    """
    meta = OrderedDict()
    if reward_free:
        # mode_key, skill_key = jax.random.split(key)
        skill_key = key
        # mode = bool(jax.random.bernoulli(key=mode_key, p=0.4))
        mode = episode_partition_mode_selector(time_step=time_step, partitions=partitions)
        if skill_mode == 'half':
            first_half_dim = int(skill_dim / 2)
            second_half_dim = skill_dim - first_half_dim
            zero = jnp.zeros(shape=(first_half_dim,), dtype=jnp.float32)
            uniform = jax.random.uniform(skill_key, shape=(second_half_dim,), minval=0., maxval=1.)
            if mode:
                skill = jnp.concatenate([zero, uniform])
            else:
                skill = jnp.concatenate([uniform, zero])
        elif skill_mode == 'sign':
            sign = -1. if mode else 1.
            skill = jax.random.uniform(skill_key, shape=(skill_dim,), minval=0., maxval=1.) * sign
        elif skill_mode == 'same':
            skill = jax.random.uniform(skill_key, shape=(skill_dim,), minval=0., maxval=1.)
        elif skill_mode == 'discrete':
            sign = -1. if mode else 1.
            skill = jnp.ones((skill_dim,)) * sign
            # sign = 0. if mode else 1.
            # skill = jnp.ones(shape=(skill_dim,), dtype=jnp.float32) * sign
        meta['mode'] = mode
    else:
        # outputs best skill after exploration loop
        # use constant skill function for baseline
        if search_mode == 'random_grid_search':
            skill = calculations.skill_utils.random_grid_search_skill(
                skill_dim=skill_dim,
                global_timestep=step,
                skill_tracker=skill_tracker,
                key=key
            )
        elif search_mode == 'grid_search':
            skill = calculations.skill_utils.grid_search_skill(
                skill_dim=skill_dim,
                global_timestep=step,
                skill_tracker=skill_tracker,
            )
        elif search_mode == 'random_search':
            skill = calculations.skill_utils.random_search_skill(
                skill_dim=skill_dim,
                global_timestep=step,
                skill_tracker=skill_tracker,
                key=key
            )
        elif search_mode == 'constant':
            skill = calculations.skill_utils.constant_fixed_skill(
                skill_dim=skill_dim,
            )
        elif search_mode == 'explore':
            skill = jnp.ones((skill_dim,))
            
        elif search_mode == 'control':
            skill = -jnp.ones((skill_dim,))

        if skill_tracker.update:
            # first step
            if skill_tracker.score_step == 0:
                pass
            elif skill_tracker.score_sum / skill_tracker.score_step > skill_tracker.best_score:
                skill_tracker = skill_tracker._replace(
                    best_skill=skill_tracker.current_skill,
                    best_score=skill_tracker.score_sum / skill_tracker.score_step
                )
            skill_tracker = skill_tracker._replace(
                score_sum=0.,
                score_step=0
            )
        # skill = jnp.ones(skill_dim, dtype=jnp.float32) * 0.5
        skill_tracker = skill_tracker._replace(current_skill=skill)

    meta['skill'] = skill

    return meta, skill_tracker

class DDPGAgentMultiModalSkill(agents.DDPGAgentSkill):

    """Implement DDPG with skills"""
    def __init__(self,
                 skills_cfg,
                 reward_free: bool,
                 search_mode,
                 skill_mode,
                 partitions,
                 **kwargs
    ):
        super().__init__(
            skills_cfg,
            reward_free,
            **kwargs
        )
        # init in exploration mode
        self._mode = bool(0)

        to_jit = jax.jit if kwargs['to_jit'] else lambda x: x

        self.get_meta_specs = partial(
            get_meta_specs, skill_dim=skills_cfg.skill_dim, reward_free=reward_free
        )
        self.init_meta = partial(
            init_meta,
            partitions=partitions,
            reward_free=reward_free,
            skill_dim=skills_cfg.skill_dim,
            search_mode=search_mode,
            skill_mode=skill_mode
        )

    def update_meta(self,
                    key: jax.random.PRNGKey,
                    meta: OrderedDict,
                    step: int,
                    update_skill_every: int,
                    time_step,
                    skill_tracker=None,
                ) -> Tuple[OrderedDict, Any]:
        if step % update_skill_every == 0:
            return self.init_meta(key, step=step, skill_tracker=skill_tracker, time_step=time_step)
        return meta, skill_tracker