from typing import List

import hydra

from .intrinsic_reward_base import IntrinsicReward
from .cic import CICReward
from .multimodal_cic import MultimodalCICReward

def make_intrinsic_reward(cfg):
    return hydra.utils.instantiate(cfg)