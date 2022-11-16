from typing import NamedTuple

import dm_env

from .dmc import make as make_dmc_env


def make_env(action_type: str, cfg: NamedTuple, seed: int) -> dm_env.Environment:
    if action_type == 'continuous':
        return make_dmc_env(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, seed)
    raise NotImplementedError
