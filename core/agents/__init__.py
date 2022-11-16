from typing import Tuple, NamedTuple

import hydra

from .agent_base import Agent
from .ddpg import DDPGAgent, DDPGTrainState
from .ddpg_skill import DDPGAgentSkill
from .ddpg_multimodal_skill_torch import DDPGAgentMultiModalSkill


def make_agent(obs_type, action_shape, agent_cfg):
    if agent_cfg.action_type == 'continuous':
        return make_continuous_agent(action_shape, agent_cfg)
    elif agent_cfg.action_type == 'discrete':
        return make_discrete_agent(obs_type, action_shape, agent_cfg)
    else:
        raise NotImplementedError

def make_continuous_agent(action_shape, agent_cfg):
    agent_cfg.network_cfg.action_shape  = action_shape
    return hydra.utils.instantiate(agent_cfg)

def make_discrete_agent(obs_type: str, action_shape: Tuple[int], cfg):
    cfg.network_cfg.obs_type = obs_type
    cfg.network_cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)
