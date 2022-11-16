import torch
import torch.nn.functional as F
import torch.nn as nn
import jax
import jax.numpy as jnp


def _random_flip_single_image(image, rng):
  _, flip_rng = jax.random.split(rng)
  should_flip_lr = jax.random.uniform(flip_rng, shape=()) <= 0.5
  image = jax.lax.cond(should_flip_lr, image, jnp.fliplr, image, lambda x: x)
  return image


def random_flip(images, rng):
  rngs = jax.random.split(rng, images.shape[0])
  return jax.vmap(_random_flip_single_image)(images, rngs)


def random_shift_aug(x: jnp.ndarray):
    """x: [N, H, W, C]"""
    x = x.astype(dtype=jnp.float32)
    n, h, w, c = x.shape
    assert h == w

    return jax.lax.stop_gradient(x)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)