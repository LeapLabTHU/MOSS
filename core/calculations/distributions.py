import jax
import jax.numpy as jnp


class Distribution:
    """
    Abstract base class for probability distribution
    """
    def __init__(self, batch_shape, event_shape):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

    def sample(self, sample_shape):
        pass

class TruncNormal:
    def __init__(self, loc, scale, low=-1.0, high=1.0):
        """ Trunc from -1 to 1 for DMC action space
        :param loc: mean (N, action_dim)
        :param scale: stddev ()
        :param low: clamp to low
        :param high: clamp to high
        """
        self.low = low
        self.high = high
        self.loc = loc
        self.scale = scale
        self.eps = 1e-6

    def mean(self):
        return self.loc

    def sample(self,
             clip=None,
             *,
             seed: jax.random.PRNGKey,
             # sample_shape: Sequence[int] = (),
    ):
        """Samples an event.

            Args:
              clip: implements clipped noise in DrQ-v2
              seed: PRNG key or integer seed.

            Returns:
              A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
            """
        sample_shape = self.loc.shape
        noise = jax.random.normal(seed, sample_shape) # has to be same shape as loc which specifies the mean for each individual Gaussians
        noise *= self.scale

        if clip is not None:
            # clip N(0, var) of exploration schedule in DrQ-v2
            noise = jnp.clip(noise, a_min=-clip, a_max=clip)
        x = self.loc + noise
        # return jnp.clip(x, a_min=self.low, a_max=self.high)
        clamped_x = jnp.clip(x, a_min=self.low + self.eps, a_max=self.high - self.eps)
        x = x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped_x) # trick to backprop on x without clamping affecting it
        return x
#
# class TruncNormal(distrax.Normal):
#     def __init__(self, loc, scale, low=-1.0, high=1.0):
#         """ Trunc from -1 to 1 for DMC action space
#         :param loc: mean
#         :param scale: stddev
#         :param low:
#         :param high:
#         :param eps:
#         """
#         super(TruncNormal, self).__init__(loc=loc, scale=scale)
#
#         self.low = low
#         self.high = high
#         # self.eps = eps
#
#     def _clamp(self, x):
#         """ Clamping method for TruncNormal"""
#         clamped_x = jnp.clip(x, self.low, self.high)
#         x = x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped_x)
#         return x
#
#     def sample(self,
#              clip=None,
#              *,
#              seed, #: Union[IntLike, PRNGKey],
#              sample_shape = (),#: Union[IntLike, Sequence[IntLike]] = ()
#     ):
#         """Samples an event.
#
#             Args:
#               clip: implements clipped noise in DrQ-v2
#               seed: PRNG key or integer seed.
#               sample_shape: Additional leading dimensions for sample.
#
#             Returns:
#               A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
#             """
#         # this line check if rng is a PRNG key and sample_shape a tuple if not it converts them.
#         # rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
#         num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product
#
#         eps = self._sample_from_std_normal(seed, num_samples)
#         scale = jnp.expand_dims(self._scale, range(eps.ndim - self._scale.ndim))
#         loc = jnp.expand_dims(self._loc, range(eps.ndim - self._loc.ndim))
#
#         eps *= scale
#         if clip is not None:
#             # clip N(0, var) of exploration schedule in DrQ-v2
#             eps = jnp.clip(eps, a_min=-clip, a_max=clip)
#         samples = loc + eps
#         samples = self._clamp(samples)
#         return samples.reshape(sample_shape + samples.shape[1:])

#
# import torch
# from torch import distributions as pyd
# from torch.distributions.utils import _standard_normal
#
#
# class TruncatedNormal(pyd.Normal):
#     def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
#         super().__init__(loc, scale, validate_args=False)
#         self.low = low
#         self.high = high
#         self.eps = eps
#
#     def _clamp(self, x):
#         clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
#         x = x - x.detach() + clamped_x.detach()
#         return x
#
#     def sample(self, clip=None, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape)
#         eps = _standard_normal(shape,
#                                dtype=self.loc.dtype,
#                                device=self.loc.device)
#         eps *= self.scale
#         if clip is not None:
#             eps = torch.clamp(eps, -clip, clip)
#         x = self.loc + eps
#         return self._clamp(x)
#
# if __name__ == "__main__":
#     truncNormal = TruncNormal(jnp.ones((3,)), 1.)
#     samples_jax = truncNormal.sample(clip=2, seed=jax.random.PRNGKey(666))
#
#     torchtruncNormal = TruncatedNormal(torch.ones(3), 1.)
#     samples_torch = torchtruncNormal.sample(clip=2)
#
#     print(samples_jax, samples_torch)
    # [[0.96648777  1.]
    #  [0.4025777   1.]
    # [-0.59399736
    # 1.]]