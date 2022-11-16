from typing import NamedTuple, Callable, Mapping, Union, Any, Tuple, Text
from functools import partial
from collections import OrderedDict
import logging

import optax
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import dm_env

from core import agents
from core import calculations
from core import data

LogsDict = Mapping[Text, jnp.ndarray]
FloatStrOrBool = Union[str, float, bool]


class Actor(hk.Module):
    """The actor in DDPG is the policy gradient"""

    def __init__(self,
                 obs_type: str,
                 action_dim: int,
                 feature_dim: int,
                 hidden_dim: int,
                 ln_config: Mapping[str, FloatStrOrBool],
                 ):
        super(Actor, self).__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = calculations.trunk(ln_config, feature_dim)
        policy_layers = [calculations.linear_relu(hidden_dim)]
        if obs_type == 'pixels':
            policy_layers.append(calculations.linear_relu(hidden_dim))
        policy_layers.append(hk.Linear(action_dim, w_init=calculations.default_linear_init))
        self.policy = hk.Sequential(policy_layers)

    def __call__(self, features: jnp.ndarray, std: float):
        features = self.trunk(features)
        mu = jax.nn.tanh(self.policy(features))  # action_dim
        std = jnp.ones_like(mu) * std
        return calculations.TruncNormal(loc=mu, scale=std)


class Critic(hk.Module):
    """The Q function"""

    def __init__(self,
                 obs_type: str,
                 feature_dim: int,
                 hidden_dim: int,
                 ln_config: Mapping[str, FloatStrOrBool]
                 ):
        super(Critic, self).__init__()

        self.obs_type = obs_type
        if obs_type == 'pixels':
            self.trunk = calculations.trunk(ln_config, feature_dim)
        else:
            self.trunk = calculations.trunk(ln_config, hidden_dim)

        def make_q(name):
            q_layers = [calculations.linear_relu(hidden_dim)]
            if obs_type == 'pixels':
                q_layers.append(calculations.linear_relu(hidden_dim))
            q_layers.append(hk.Linear(1, w_init=calculations.default_linear_init))
            return hk.Sequential(q_layers, name=name)

        self.Q1 = make_q('q1')
        self.Q2 = make_q('q2')

    def __call__(self, features: jnp.ndarray, action: jnp.ndarray):
        # for states actions come in the beginning
        x = features if self.obs_type == 'pixels' else jnp.concatenate([features, action], axis=-1)

        x = self.trunk(x)
        # for pixels actions will be added after trunk
        x = jnp.concatenate([x, action], axis=-1) if self.obs_type == 'pixels' else x

        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


def ddpg_act(features: jnp.ndarray,
             std: float,
             network_cfg: Mapping[str, Any]
             ) -> calculations.TruncNormal:
    obs_type: str = network_cfg['obs_type']
    ln_config: Mapping[str, FloatStrOrBool] = network_cfg['ln_config']
    action_dim: int = network_cfg['action_shape'][0]
    feature_dim: int = network_cfg['feature_dim']
    hidden_dim: int = network_cfg['hidden_dim']
    actor: Callable = Actor(obs_type,
                            action_dim=action_dim,
                            feature_dim=feature_dim,
                            hidden_dim=hidden_dim,
                            ln_config=ln_config)

    dist = actor(features=features, std=std)
    return dist


def ddpg_critic(features: jnp.ndarray,
                action: jnp.ndarray,
                network_cfg: Mapping[str, Any]
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    obs_type: str = network_cfg['obs_type']
    ln_config: Mapping[str, FloatStrOrBool] = network_cfg['ln_config']
    feature_dim: int = network_cfg['feature_dim']
    hidden_dim: int = network_cfg['hidden_dim']
    critic: Callable = Critic(obs_type=obs_type,
                              feature_dim=feature_dim,
                              hidden_dim=hidden_dim,
                              ln_config=ln_config)
    q1, q2 = critic(features, action)
    return q1, q2


class DDPGTrainState(NamedTuple):
    actor_params: hk.Params
    critic_params: hk.Params
    critic_target_params: hk.Params
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState


class DDPGAgent(agents.Agent):
    """Implement DDPG"""

    def __init__(self,
                 to_jit: bool,
                 stddev_schedule: float,
                 stddev_clip: float,
                 lr_actor: float,
                 lr_critic: float,
                 l2_weight: float,
                 critic_target_tau: float,
                 network_cfg: DictConfig,
                 replay_buffer_cfg: DictConfig,
                 **kwargs
    ):
        """
        Implements DrQ-v2
        :param to_jit: whether to jit update and act functions
        :param stddev_schedule: stddev schedule for noise, a constant here
        :param stddev_clip: stddev clipping value
        :param lr_actor: learning rate of actor
        :param lr_critic: learning rate of critic
        :param critic_target_tau: moving average hyper parameter of critic target
        :param network_cfg: configs for critic and actor neural network
        """
        self.stddev_schedule = partial(calculations.schedule, schdl=stddev_schedule)
        self._update_critic_target = partial(
            calculations.polyak_averaging, tau=critic_target_tau
        )
        self._actor_loss = partial(
            self._actor_loss, stddev_clip=stddev_clip, l2_weight=l2_weight
        )
        self._compute_target = partial(
            self._compute_target, stddev_clip=stddev_clip
        )

        # if state then input is obs_dim + skill_dim
        self.actor = hk.without_apply_rng(hk.transform(
            partial(ddpg_act, network_cfg=network_cfg)
        )
        )
        self.critic = hk.without_apply_rng(hk.transform(
            partial(ddpg_critic, network_cfg=network_cfg)
        )
        )

        self.actor_opt = optax.adam(learning_rate=lr_actor)
        self.critic_opt = optax.adam(learning_rate=lr_critic)

        # jit act and update
        to_jit_fct = jax.jit if to_jit else lambda x: x
        self._act_sample = to_jit_fct(partial(
            self._act, greedy=False
        ))
        self._act_greedy = to_jit_fct(partial(
            self._act, greedy=True
        ))
        self.update = to_jit_fct(self.update)


        self.init_replay_buffer = partial(
            self.init_replay_buffer,
            replay_buffer_cfg=replay_buffer_cfg
        )
        self.init_params = partial(
            self.init_params,
            obs_type=network_cfg.obs_type
        )
        # self.select_action = partial(
        #     self.select_action,
        #     action_dim=network_cfg['action_shape']
        # )
        self._adder = None
        self._data_iterator = None
        self._loader = None
        # self._client = None
        # self._server = None

    def init_replay_buffer(
        self,
        replay_buffer_cfg,
        environment_spec,
        **kwargs,
    ):

        if self._adder is None:
            self._adder = data.ReplayBufferStorage(data_specs=environment_spec,
                                                   meta_specs=self.get_meta_specs(),
                                                   replay_dir=kwargs['replay_dir'])
        if self._data_iterator is None:
            self._loader = data.make_replay_loader(self._adder,
                                                   replay_buffer_cfg.replay_buffer_size,
                                                   replay_buffer_cfg.batch_size,
                                                   replay_buffer_cfg.num_workers,
                                                   replay_buffer_cfg.nstep,
                                                   replay_buffer_cfg.discount,
                                                   self.get_meta_specs()
                                                   )

    def store_timestep(self,
                       time_step: dm_env.TimeStep,
                       meta
                       ):

        self._adder.add(time_step, meta)


    def sample_timesteps(self):
        if self._data_iterator is None:
            self._data_iterator = iter(self._loader)
        return next(self._data_iterator)
        # return next(self._data_iterator).data

    def __len__(self):
        return len(self._adder)

    def init_params(self,
                    init_key: jax.random.PRNGKey,
                    dummy_obs: jnp.ndarray,
                    summarize: bool = False,
                    obs_type: str = 'states',
                    checkpoint_state: DDPGTrainState = None
                    ):
        if obs_type == 'pixels':
            dummy_obs = dummy_obs[None]
        # init hk.Params
        dummy_std = 0.3
        actor_init = self.actor.init(rng=init_key, features=dummy_obs, std=dummy_std)
        dummy_distribution = self.actor.apply(params=actor_init, features=dummy_obs, std=dummy_std)
        dummy_action = dummy_distribution.sample(seed=init_key)
        critic_init = self.critic.init(rng=init_key, features=dummy_obs, action=dummy_action)
        critic_target_init = self.critic.init(rng=init_key, features=dummy_obs,
                                              action=dummy_action)  # different init for critic and target?
        logger = logging.getLogger(__name__)
        if summarize:
            logger.info(hk.experimental.tabulate(self.actor)(dummy_obs, dummy_std))
            logger.info(hk.experimental.tabulate(self.critic)(dummy_obs, dummy_action))
        # init optax.OptState
        actor_opt_init = self.actor_opt.init(actor_init)
        critic_opt_init = self.critic_opt.init(critic_init)

        if checkpoint_state is not None:
            logger.info("Initializing state from checkpoint")
            # load trunk of critic
            critic_init['critic/~/trunk_linear'] = checkpoint_state.critic_params['critic/~/trunk_linear']
            critic_init['critic/~/trunk_ln'] = checkpoint_state.critic_params['critic/~/trunk_ln']
            return DDPGTrainState(
                actor_params=checkpoint_state.actor_params,
                critic_params=critic_init,
                critic_target_params=critic_target_init,
                actor_opt_state=actor_opt_init,
                critic_opt_state=critic_opt_init,
            )
        else:
            logger.info("No checkpoint given, training from scratch")
            return DDPGTrainState(
                actor_params=actor_init,
                critic_params=critic_init,
                critic_target_params=critic_target_init,
                actor_opt_state=actor_opt_init,
                critic_opt_state=critic_opt_init
            )

    def select_action(self,
        obs:jnp.ndarray,
        meta: OrderedDict,
        step: int,
        key: jax.random.PRNGKey,
        greedy: bool,
        state: DDPGTrainState
    ):

        if greedy:
            return np.array(self._act_greedy(state, obs, meta, step, key)[0])
        return np.array(self._act_sample(state, obs, meta, step, key)[0])

    def _act(self,
            state: DDPGTrainState,
            obs: jnp.ndarray,
            meta: OrderedDict,
            step: int,
            key: jax.random.PRNGKey,
            greedy: bool,
    ) -> calculations.TruncNormal:
        """return actions of actor [1, action_dim] """
        # expand to include batch dimension
        features = self._get_skill_obs(obs[None], meta)
        stddev = self.stddev_schedule(step=step)
        dist = self.actor.apply(state.actor_params, features=features, std=stddev)
        if greedy:
            return dist.mean()
        else:
            return dist.sample(clip=None, seed=key)

    def _actor_loss(self,
                    actor_params: hk.Params,
                    critic_params: hk.Params,
                    key: jax.random.PRNGKey,
                    obs: jnp.ndarray,
                    stddev: float,
                    stddev_clip: float,
                    l2_weight: float
                    ) -> Tuple[jnp.ndarray, LogsDict]:
        dist = self.actor.apply(actor_params, features=obs, std=stddev)
        action = dist.sample(clip=stddev_clip, seed=key)
        Q1, Q2, = self.critic.apply(critic_params, obs, action)
        Q = jnp.minimum(Q1, Q2)
        actor_loss = -jnp.mean(Q)
        actor_loss += calculations.l2_loss_without_bias(actor_params) * l2_weight
        logs = dict(
            actor_loss=-actor_loss,
        )
        return actor_loss, logs

    def _critic_loss(self,
                     critic_params: hk.Params,
                     obs: jnp.ndarray,
                     buffer_action: jnp.ndarray,
                     target: jnp.ndarray
                     ) -> Tuple[jnp.ndarray, LogsDict]:
        Q1, Q2 = self.critic.apply(critic_params, obs, buffer_action)
        loss = jnp.mean(jnp.square(Q1 - target)) + jnp.mean(jnp.square(Q2 - target))
        logs = dict(
            critic_loss=loss,
            critic_q1=jnp.mean(Q1),
            critic_q2=jnp.mean(Q2),
            critic_q_target=jnp.mean(target)
        )
        return loss, logs

    def _update_actor(self,
                      actor_params: hk.Params,
                      critic_params: hk.Params,
                      actor_opt_state: optax.OptState,
                      key: jax.random.PRNGKey,
                      obs: jnp.ndarray,
                      step: int
                      ) -> Tuple[Any, LogsDict]:
        """The actor maximize the expected value of the Q(s, actor)"""
        grad_fn = jax.grad(self._actor_loss, argnums=0, has_aux=True)
        stddev = self.stddev_schedule(step=step)
        grads, logs = grad_fn(actor_params, critic_params, key, obs, stddev)
        deltas, actor_opt_state = self.actor_opt.update(grads, actor_opt_state)
        actor_params = optax.apply_updates(params=actor_params, updates=deltas)
        return (actor_params, actor_opt_state), logs

    def _compute_target(self,
                        actor_params: hk.Params,
                        critic_target_params: hk.Params,
                        key: jax.random.PRNGKey,
                        next_obs: jnp.ndarray,
                        discount: jnp.ndarray,
                        reward: jnp.ndarray,
                        step: int,
                        stddev_clip
                        ):
        """Clipped Double Q-learning y = r + min(Q(s+1, a+1))"""
        stddev = self.stddev_schedule(step=step)
        dist = self.actor.apply(actor_params, next_obs, std=stddev)
        next_action = dist.sample(clip=stddev_clip, seed=key)
        target_Q1, target_Q2 = self.critic.apply(critic_target_params, next_obs, next_action)
        target_V = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + (discount * target_V)
        return target_Q

    def _update_critic(self,
                       critic_params: hk.Params,
                       critic_opt_state: optax.OptState,
                       target_Q: jnp.ndarray,
                       obs: jnp.ndarray,
                       buffer_action: jnp.ndarray,
    ) -> Tuple[Any, LogsDict]:

        grad_fn = jax.grad(self._critic_loss, argnums=0, has_aux=True)
        critic_grads, logs = grad_fn(critic_params, obs, buffer_action, target_Q)
        critic_deltas, critic_opt_state = self.critic_opt.update(critic_grads, critic_opt_state)
        critic_params = optax.apply_updates(params=critic_params, updates=critic_deltas)

        return (critic_params, critic_opt_state), logs


    def _get_skill_obs(self, obs, meta):
        features = [obs]
        for key, value in meta.items(): # empty meta does not enter this
            # only concatenate skill, do not concat other metas
            if key == 'skill':
                # expand batch dim if not enough dimensions
                features.append(value[None] if value.ndim < obs.ndim else value)
        features = jnp.concatenate(features, axis=-1)
        return features

    def update(self,
               state: DDPGTrainState,
               key: jax.random.PRNGKey,
               batch: data.Batch,
               step: int
    ):
        """
        batch dim [bs, c], [bs, nb_action], [bs, 1], [bs, 1]
        """
        obs = batch.observation
        action = batch.action
        reward = batch.reward
        discount = batch.discount
        next_obs = batch.next_observation
        meta = batch.extras

        obs = self._get_skill_obs(obs, meta)
        next_obs = self._get_skill_obs(next_obs, meta)

        #TODO randshit aug + bilinear interpolation for pixels https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
        logs = dict()
        logs['batch_reward'] = jnp.array(reward).mean()

        step_key, key = tuple(jax.random.split(key, num=2))

        (actor_params, actor_opt_state), logs_actor = self._update_actor(state.actor_params,
                                                                         state.critic_params,
                                                                         state.actor_opt_state,
                                                                         step_key, obs, step=step)

        step_key, key = tuple(jax.random.split(key, num=2))
        critic_target = self._compute_target(state.actor_params,
                                             state.critic_target_params,
                                             step_key, next_obs, discount, reward, step=step)

        (critic_params, critic_opt_state), logs_critic = \
            self._update_critic(state.critic_params,
                                state.critic_opt_state,
                                critic_target, obs, action)

        logs.update(logs_actor)
        logs.update(logs_critic)

        # update critic target
        critic_target_params = self._update_critic_target(params=critic_params,
                                                          target_params=state.critic_target_params)

        return DDPGTrainState(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            critic_target_params=critic_target_params,
        ), logs

    def get_meta_specs(self, *args, **kwargs):
        return tuple()

    def init_meta(self, *args, **kwargs):
        return OrderedDict()

    def update_meta(self, *args, **kwargs):
        return OrderedDict()
