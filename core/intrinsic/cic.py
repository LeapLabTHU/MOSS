from typing import Callable, Mapping, Any, NamedTuple, Tuple, Dict, Union
from functools import partial
import logging

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from core import intrinsic
from core import calculations
from core import data

class CICnetwork(hk.Module):
    def __init__(self,
                 hidden_dim: int,
                 skill_dim: int,
                 project_skill: bool
                 ):
        super(CICnetwork, self).__init__()

        self.state_net = calculations.mlp(hidden_dim, skill_dim, name='state_net')
        # self.state_net = calculations.mlp(hidden_dim, skill_dim//2, name='state_net')
        # self.next_state_net = calculations.mlp(hidden_dim, skill_dim, name='next_state_net')
        self.pred_net = calculations.mlp(hidden_dim, skill_dim, name='pred_net')

        if project_skill:
            self.skill_net = calculations.mlp(hidden_dim, skill_dim, name='skill_net')
        else:
            self.skill_net = calculations.Identity()

    def __call__(self, state, next_state, skill, is_training=True): # input is obs_dim - skill_dim
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        # next_state = self.next_state_net(next_state)
        if is_training:
            query = self.skill_net(skill)
            key = self.pred_net(jnp.concatenate([state, next_state], axis=-1))
            return query, key
        else:
            return state, next_state

def cic_foward(state: jnp.ndarray,
               next_state: jnp.ndarray,
               skill: jnp.ndarray,
               is_training: bool,
               network_cfg: Mapping[str, Any]
):
    model: Callable = CICnetwork(
        hidden_dim=network_cfg['hidden_dim'],
        skill_dim=network_cfg['skill_dim'],
        project_skill=network_cfg['project_skill']
    )
    return model(state, next_state, skill, is_training)


class CICState(NamedTuple):
    cic_params: hk.Params
    cic_opt_params: optax.OptState
    running_mean: Union[float, None]
    running_std: Union[float, None]
    running_num: Union[float, None]


class CICReward(intrinsic.IntrinsicReward):
    def __init__(self,
                 to_jit: bool,
                 network_cfg: Mapping[str, Any],
                 lr: float,
                 knn_entropy_config,
                 temperature: float,
                 name: str = 'cic',
                 ):
        self.cic = hk.without_apply_rng(
            hk.transform(
                partial(
                    cic_foward,
                    network_cfg=network_cfg,
                )
            )
        )
        self._cpc_loss = partial(self._cpc_loss, temperature=temperature)
        self.init_params = partial(self.init_params, skill_dim=network_cfg['skill_dim'])
        self.cic_optimizer = optax.adam(learning_rate=lr)
        self.entropy_estimator = partial(calculations.particle_based_entropy,
                                         **knn_entropy_config)
        if to_jit:
            self.update_batch = jax.jit(self.update_batch)

    def init_params(self,
                    init_key: jax.random.PRNGKey,
                    dummy_obs: jnp.ndarray,
                    skill_dim: int,
                    summarize: bool = True
    ):
        # batch_size = dummy_obs.shape[0]
        dummy_skill = jax.random.uniform(key=init_key, shape=(skill_dim, ), minval=0, maxval=1)
        cic_init = self.cic.init(rng=init_key, state=dummy_obs, next_state=dummy_obs, skill=dummy_skill, is_training=True)
        cic_opt_init = self.cic_optimizer.init(cic_init)
        if summarize:
            logger = logging.getLogger(__name__)
            summarize_cic_forward = partial(self.cic.apply, is_training=True) # somehow only works this way
            logger.info(hk.experimental.tabulate(summarize_cic_forward)(cic_init, dummy_obs, dummy_obs, dummy_skill))
        return CICState(
            cic_params=cic_init,
            cic_opt_params=cic_opt_init,
            running_mean=jnp.zeros((1,)),
            running_std=jnp.ones((1,)),
            running_num=1e-4
        )

    def _cpc_loss(self,
                  cic_params: hk.Params,
                  obs: jnp.ndarray,
                  next_obs: jnp.ndarray,
                  skill: jnp.ndarray,
                  temperature: float
    ):
        query, key = self.cic.apply(cic_params, obs, next_obs, skill, is_training=True) #(b, c)
        # loss = calculations.noise_contrastive_loss(query, key, temperature=temperature)
        loss = calculations.cpc_loss(query=query, key=key)
        logs = dict(
            cpc_loss=loss
        )
        return loss, logs
        # return noise_contrastive_loss(query, key)

    def _update_cic(self,
                    cic_params: hk.Params,
                    cic_opt_params: optax.OptState,
                    obs: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    skill: jnp.ndarray
    ):
        grad_fn = jax.grad(self._cpc_loss, has_aux=True)
        grads, logs = grad_fn(cic_params, obs, next_obs, skill)
        deltas, cic_opt_params = self.cic_optimizer.update(grads, cic_opt_params)
        cic_params = optax.apply_updates(cic_params, deltas)
        return (cic_params, cic_opt_params), logs

    def compute_reward(self, cic_params, obs, next_obs, skill, running_mean, running_std, running_num):
        source, target = self.cic.apply(cic_params, obs, next_obs, skill, is_training=False)
        reward, running_mean, running_std, running_num = self.entropy_estimator(
                                                                source=source,
                                                                target=target,
                                                                num=running_num,
                                                                mean=running_mean,
                                                                std=running_std)
        return reward, running_mean, running_std, running_num

    def update_batch(self,
               state: CICState,
               batch: data.Batch,
               step: int,
    ) -> Tuple[CICState, NamedTuple, Dict]:
        obs = batch.observation
        extrinsic_reward = batch.reward
        next_obs = batch.next_observation
        meta = batch.extras
        skill = meta['skill']
        """ Updates CIC and batch"""
        logs = dict()
        # TODO add aug for pixel based
        (cic_params, cic_opt_params), cic_logs = self._update_cic(
                                                            cic_params=state.cic_params,
                                                            cic_opt_params=state.cic_opt_params,
                                                            obs=obs,
                                                            next_obs=next_obs,
                                                            skill=skill)
        logs.update(cic_logs)

        intrinsic_reward, running_mean, running_std, running_num = self.compute_reward(
                                                            cic_params=state.cic_params,
                                                            obs=obs,
                                                            next_obs=next_obs,
                                                            running_num=state.running_num,
                                                            skill=skill,
                                                            running_mean=state.running_mean,
                                                            running_std=state.running_std)

        logs['intrinsic_reward'] = jnp.mean(intrinsic_reward)
        logs['extrinsic_reward'] = jnp.mean(extrinsic_reward)

        return CICState(
            cic_params=cic_params,
            cic_opt_params=cic_opt_params,
            running_mean=running_mean,
            running_std=running_std,
            running_num=running_num
        ), batch._replace(reward=intrinsic_reward), logs
