from typing import NamedTuple, Tuple, Dict

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

from core import intrinsic
from core import data


class MultimodalCICState(NamedTuple):
    cic_params: hk.Params
    cic_opt_params: optax.OptState
    mode_0_running_mean: jnp.ndarray
    mode_0_running_std: jnp.ndarray
    mode_0_running_num: float


class RunningStatistics(NamedTuple):
    mode_0_running_mean: jnp.ndarray
    mode_0_running_std: jnp.ndarray
    mode_0_running_num: float


class MultimodalCICReward(intrinsic.CICReward):
    def __init__(self, to_jit, *args, **kwargs):
        # only jit the update fn
        super().__init__(False, *args, **kwargs)
        # # rewrite or will inherent
        if to_jit:
            self._update_cic = jax.jit(self._update_cic)
            self.entropy_estimator = jax.jit(self.entropy_estimator)

    def init_params(self,
                    init_key: jax.random.PRNGKey,
                    dummy_obs: jnp.ndarray,
                    skill_dim: int,
                    summarize: bool = True,
    ):
        cic_state = super().init_params(init_key, dummy_obs, skill_dim, summarize)
        return MultimodalCICState(
            cic_params=cic_state.cic_params,
            cic_opt_params=cic_state.cic_opt_params,
            mode_0_running_mean=jnp.zeros((1,)),
            mode_0_running_std=jnp.ones((1,)),
            mode_0_running_num=1e-4,
        )

    def compute_reward(self,
                       cic_params,
                       obs,
                       next_obs,
                       skill,
                       statistics,
                       **kwargs
    ):
        source_0, target_0 = self.cic.apply(cic_params,
                                            obs,
                                            next_obs,
                                            skill,
                                            is_training=False)
        reward, running_mean_0, running_std_0, running_num_0 = self.entropy_estimator(
            source=source_0,
            target=target_0,
            mean=statistics.mode_0_running_mean,
            std=statistics.mode_0_running_std,
            num=statistics.mode_0_running_num,
        )

        return reward, RunningStatistics(
            running_mean_0,
            running_std_0,
            running_num_0,
        )

    def update_batch(self,
               state: MultimodalCICState,
               batch: data.Batch,
               step: int,
    ) -> Tuple[MultimodalCICState, data.Batch, Dict]:
        """ Updates CIC and batch"""
        obs = batch.observation
        extrinsic_reward = batch.reward
        next_obs = batch.next_observation
        meta = batch.extras
        skill = meta['skill']
        logs = dict()
        # TODO add aug for pixel baseds
        (cic_params, cic_opt_params), cic_logs = self._update_cic(
                                                cic_params=state.cic_params,
                                                cic_opt_params=state.cic_opt_params,
                                                obs=jnp.concatenate(obs),
                                                next_obs=jnp.concatenate(next_obs),
                                                skill=jnp.concatenate(skill))
        logs.update(cic_logs)

        intrinsic_reward, statistics = self.compute_reward(cic_params=state.cic_params,
                                                       obs=jnp.concatenate(obs),
                                                       next_obs=jnp.concatenate(next_obs),
                                                       skill=jnp.concatenate(meta['skill']),
                                                       statistics=state)
        # todo do we care about logging? put before to prevent moving out of gpu and putting back
        logs['intrinsic_reward'] = jnp.mean(intrinsic_reward)
        logs['extrinsic_reward'] = jnp.mean(jnp.concatenate(extrinsic_reward))  # don't mean on a list

        intrinsic_reward = np.array(intrinsic_reward)
        intrinsic_reward[len(obs[0]):, :] *= -1


        return MultimodalCICState(
            cic_params=cic_params,
            cic_opt_params=cic_opt_params,
            mode_0_running_mean=statistics.mode_0_running_mean,
            mode_0_running_std=statistics.mode_0_running_std,
            mode_0_running_num=statistics.mode_0_running_num,
        ), data.Batch(
            observation=jnp.concatenate(obs),
            action=jnp.concatenate(batch.action),
            reward=intrinsic_reward,
            discount=jnp.concatenate(batch.discount),
            next_observation=jnp.concatenate(next_obs),
            extras=dict(skill=jnp.concatenate(skill))
        ), logs
