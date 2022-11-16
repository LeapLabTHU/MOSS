from typing import NamedTuple, Any, Dict
import time

import wandb
import haiku as hk

from core.calculations import skill_utils


def log_params_to_wandb(params: hk.Params, step: int):
    if params:
        for module in sorted(params):
            if 'w' in params[module]:
                wandb.log({
                    f'{module}/w': wandb.Histogram(params[module]['w'])
                }, step=step)
            if 'b' in params[module]:
                wandb.log({
                    f'{module}/b': wandb.Histogram(params[module]['b'])
                }, step=step)


class LogParamsEvery:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, params: hk.Params, step):
        if self._every is None:
            pass
        every = self._every // self._action_repeat
        if step % every == 0:
            log_params_to_wandb(params, step)
        pass


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class CsvData(NamedTuple):
    episode_reward: float
    episode_length: int
    episode: int
    step: int
    total_time: float
    fps: float


class LoopVar(NamedTuple):
    global_step: int
    global_episode: int
    episode_step: int
    episode_reward: float
    total_reward: float
    pointer: int


class LoopsLength(NamedTuple):
    eval_until_episode: Until
    train_until_step: Until
    seed_until_step: Until
    eval_every_step: Every


def increment_step(x: LoopVar,
                   reward: float,
                   n: int = 1
                   ) -> LoopVar:
    return LoopVar(
        global_step=x.global_step + n,
        global_episode=x.global_episode,
        episode_step=x.episode_step + n,
        episode_reward=x.episode_reward + reward,
        total_reward=x.episode_reward + reward,
        pointer=x.pointer,
    )


def increment_episode(x: LoopVar,
                      n: int = 1
                      ) -> LoopVar:
    return LoopVar(
        global_step=x.global_step,
        global_episode=x.global_episode + n,
        episode_step=x.episode_step,
        episode_reward=x.episode_reward,
        total_reward=x.episode_reward,
        pointer=x.pointer,
    )


def reset_episode(x: LoopVar,
                  ) -> LoopVar:
    return LoopVar(
        global_step=x.global_step,
        global_episode=x.global_episode,
        episode_step=0,
        episode_reward=0.,
        total_reward=x.episode_reward,
        pointer=x.pointer,
    )


def update_skilltracker(
        x: skill_utils.SkillRewardTracker,
        reward: float
) -> skill_utils.SkillRewardTracker:
    # for pretrain, we dont need skill tracker
    if x is None:
        return
    return x._replace(
        score_sum=x.score_sum + reward,
        score_step=x.score_step + 1,
    )


def parse_skilltracker(
        x: skill_utils.SkillRewardTracker,
        meta: Dict[str, Any],
) -> skill_utils.SkillRewardTracker:
    if not meta or 'tracker' not in meta:
        return x
    return meta['tracker']


def init_skilltracker(
    search_steps: int,
    change_interval: int,
    low: float,
) -> skill_utils.SkillRewardTracker:
    return skill_utils.SkillRewardTracker(
        best_skill=None,
        best_score=-float('inf'),
        score_sum=0.,
        score_step=0,
        current_skill=None,
        search_steps=search_steps,
        change_interval=change_interval,
        low=low,
        update=True,
    )


def skilltracker_update_on(
        x: skill_utils.SkillRewardTracker,
) -> skill_utils.SkillRewardTracker:
    if x is None:
        return
    return x._replace(update=True)


def skilltracker_update_off(
        x: skill_utils.SkillRewardTracker,
) -> skill_utils.SkillRewardTracker:
    if x is None:
        return
    return x._replace(update=False)
