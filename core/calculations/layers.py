from typing import Callable, Mapping, Union

import haiku as hk
import jax
import jax.numpy as jnp

FloatStrOrBool = Union[str, float, bool]
default_linear_init = hk.initializers.Orthogonal()

class Identity(hk.Module):
    def __init__(self, name = 'identity'):
        super(Identity, self).__init__(name=name)

    def __call__(self, inputs):
        return inputs

def trunk(ln_config: Mapping[str, FloatStrOrBool], feature_dim: int, name='trunk') -> Callable:
    """Layer"""
    return hk.Sequential([
        hk.Linear(output_size=feature_dim, w_init=default_linear_init, name='trunk_linear'),
        hk.LayerNorm(**ln_config, name='trunk_ln'),
        jax.nn.tanh
    ], name=name)

def linear_relu(dim: int, name='linear_relu') -> Callable:
    """Layer"""
    return hk.Sequential([
        hk.Linear(output_size=dim, w_init=default_linear_init), #TODO pass it as argument
        jax.nn.relu
    ], name=name)

def mlp(dim: int, out_dim: int, name='mlp') -> Callable:
    return hk.Sequential(
        [linear_relu(dim=dim),
         linear_relu(dim=dim),
         hk.Linear(out_dim, w_init=default_linear_init)
         ],
        name=name
    )

def mlp_bottlneck(dim: int, out_dim: int, name='mlp') -> Callable:
    return hk.Sequential(
        [linear_relu(dim=dim // 2),
         linear_relu(dim=dim),
         hk.Linear(out_dim, w_init=default_linear_init)
         ],
        name=name
    )

def feature_extractor(obs: jnp.ndarray, obs_type: str, name='encoder') -> jnp.ndarray:
    """encoder"""
    if obs_type == 'pixels':
        encoder =  hk.Sequential([
            lambda x: x / 255.0 - 0.5, #FIXME put on GPU instead of CPU
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=2, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='VALID'),
            jax.nn.relu,
            hk.Flatten(preserve_dims=-3) # [N, H, W, C] -> [N, -1]
        ], name=name)
    else:
        encoder = Identity()

    return encoder(inputs=obs)


if __name__ == "__main__":
    def network(obs):
        def make_q(name):
            return hk.Sequential([
                linear_relu(10),
                hk.Linear(1, w_init=default_linear_init)
            ], name)

        q1 = make_q(name='q1')
        q2 = make_q(name='q2') # q1 neq q2
        return q1(obs), q2(obs)

    forward = hk.without_apply_rng(hk.transform(network))
    key = jax.random.PRNGKey(2)
    obs = jnp.ones((1, 10))
    state = forward.init(rng=key, obs=obs) # state = state2
    state_2 = forward.init(rng=key, obs=obs)
    print(state)