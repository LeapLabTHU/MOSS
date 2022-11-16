from typing import Tuple, Any
from collections import OrderedDict
from functools import partial

import dm_env
import jax
import jax.numpy as jnp
import numpy as np

from core import agents
from core import calculations

def init_meta(key,
              reward_free: bool,
              skill_dim: int,
              skill_tracker: calculations.skill_utils.SkillRewardTracker=None,
              step: int = None) -> Tuple[OrderedDict, Any]:

    meta = OrderedDict()
    if reward_free:
        skill = jax.random.uniform(key, shape=(skill_dim, ), minval=0., maxval=1.)

    else:
        # outputs best skill after exploration loop
        # use constant skill function for baseline
        skill = calculations.skill_utils.grid_search_skill(
            skill_dim=skill_dim,
            global_timestep=step,
            skill_tracker=skill_tracker,
        )
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


def get_meta_specs(skill_dim: int) -> Tuple:
    """
    Each element of the tuple represent one spec for a particular element
    """
    # noinspection PyRedundantParentheses
    return (dm_env.specs.Array((skill_dim,), np.float32, 'skill'),)


class DDPGAgentSkill(agents.DDPGAgent):

    """Implement DDPG with skills"""
    def __init__(self,
                 skills_cfg,
                 reward_free: bool,
                 **kwargs
    ):
        super(DDPGAgentSkill, self).__init__(**kwargs)
        self.get_meta_specs = partial(get_meta_specs, skill_dim=skills_cfg.skill_dim)
        self.init_meta = partial(
            init_meta,
            reward_free=reward_free,
            skill_dim=skills_cfg.skill_dim,
        )
        self.update_meta = partial(self.update_meta, update_skill_every=skills_cfg.update_skill_every)
        self.init_params = partial(
            self.init_params,
            obs_type=kwargs['network_cfg'].obs_type
        )

    def init_params(self,
                    init_key: jax.random.PRNGKey,
                    dummy_obs: jnp.ndarray,
                    summarize: bool = True,
                    checkpoint_state = None,
                    **kwargs
    ):
        """
        :param init_key:
        :param dummy_obs:
        :param summarize:
        :param checkpoint_state:
        :return:
        """
        skill = jnp.empty(self.get_meta_specs()[0].shape)
        dummy_obs = jnp.concatenate([dummy_obs, skill], axis=-1)
        state = super().init_params(init_key=init_key,
                                    dummy_obs=dummy_obs,
                                    summarize=summarize,
                                    checkpoint_state=checkpoint_state)
        return state

    def update_meta(self,
                    key: jax.random.PRNGKey,
                    meta: OrderedDict,
                    step: int,
                    update_skill_every: int,
                    time_step=None,
                    skill_tracker=None,
                ) -> Tuple[OrderedDict, Any]:

        if step % update_skill_every == 0:
            return self.init_meta(key, step=step, skill_tracker=skill_tracker)
        return meta, skill_tracker

