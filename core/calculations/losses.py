import math
from typing import Tuple

import jax.numpy as jnp
import jax
import torch
import chex
import haiku as hk
import tree


def l2_loss(preds: jnp.ndarray,
            targets: jnp.ndarray = None
) -> jnp.ndarray:
    """Compute l2 loss if target not provided computes l2 loss with target 0"""
    if targets is None:
        targets = jnp.zeros_like(preds)
    chex.assert_type([preds, targets], float)
    return 0.5 * (preds - targets)**2

def l2_loss_without_bias(params: hk.Params):
    l2_params = [p for ((module_name, x), p) in tree.flatten_with_path(params) if x == 'w']
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def running_stats(
    mean: jnp.ndarray,
    std: jnp.ndarray,
    x: jnp.ndarray,
    num: float,
):
    bs = x.shape[0]
    delta = jnp.mean(x, axis=0) - mean
    new_mean = mean + delta * bs / (num + bs)
    new_std = (std * num + jnp.var(x, axis=0) * bs +
      (delta**2) * num * bs / (num + bs)) / (num + bs)
    return new_mean, new_std, num + bs


def particle_based_entropy(source: jnp.ndarray,
                           target: jnp.ndarray,
                           knn_clip: float = 0.0005, # todo remove for minimization
                           knn_k: int = 16,
                           knn_avg: bool = True,
                           knn_rm: bool = True,
                           minus_mean: bool = True,
                           mean: jnp.ndarray = None,
                           std: jnp.ndarray = None,
                           num: float = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """ Implement Particle Based Entropy Estimator as in APT
    :param knn_rm:
    :param mean: mean for running mean
    :param knn_clip:
    :param knn_k: hyperparameter k
    :param knn_avg: whether to take the average over k nearest neighbors
    :param source: value to compute entropy over [b1, c]
    :param target: value to compute entropy over [b1, c]
    :return: entropy of rep # (b1, 1)
    """
    # source = target = rep #[b1, c] [b2, c]

    b1, b2 = source.shape[0], target.shape[0]
    # (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = jnp.linalg.norm(
        source[:, None, :].reshape(b1, 1, -1) - target[None, :, :].reshape(1, b2, -1),
        axis=-1,
        ord=2
    )
    # take the min of the sim_matrix to get largest=False
    reward, _ = jax.lax.top_k(
        operand=-sim_matrix, #(b1, b2)
        k=knn_k
    )
    reward = -reward

    if not knn_avg: # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if knn_rm:
            mean, std, num = running_stats(mean, std, reward, num)
            if minus_mean:
                reward = (reward - mean) / std
            else:
                reward = reward/ std
        reward = jnp.maximum(
            reward - knn_clip,
            jnp.zeros_like(reward)
        )
    else: # average over all k nearest neigbors
        reward = reward.reshape(-1, 1) #(b1 * k, 1)
        if knn_rm:
            mean, std, num = running_stats(mean, std, reward, num)
            if minus_mean:
                reward = (reward - mean) / std
            else:
                reward = reward / std
        if knn_clip >= 0.0:
            reward = jnp.maximum(
                reward - knn_clip,
                jnp.zeros_like(reward)
            )
        reward = reward.reshape((b1, knn_k))
        reward = jnp.mean(reward, axis=1, keepdims=True) # (b1, 1)

    reward = jnp.log(reward + 1.0)
    return reward, mean, std, num



def log_sum_exp(logits: jnp.ndarray):
    return jnp.log(
        jnp.sum(
            jnp.exp(logits),# [N, C]
            axis=-1
        ) # [N]
    )

def normalize(x):
    return x / (jnp.linalg.norm(x=x, ord=2, axis=-1, keepdims=True) + 1e-12)
# jnp.sqrt(jnp.sum(jnp.square(normalize(a))))

def noise_contrastive_loss(
        query,
        key,
        temperature = 0.5
):
    """
    s_i - \sum \exp s_i
    """
    query = normalize(query)
    key = normalize(key)
    logits = query @ key.T #(N, N) positive pairs on the diagonal
    logits = logits / temperature
    shifted_cov = logits - jax.lax.stop_gradient(logits.max(axis=-1, keepdims=True)) # [N, N]
    diag_indexes = jnp.arange(shifted_cov.shape[0])[:, None]# [N, 1]
    pos = jnp.take_along_axis(arr=shifted_cov, indices=diag_indexes, axis=-1) # [N, 1]
    neg = log_sum_exp(shifted_cov)
    return -jnp.mean(pos.reshape(-1) - neg.reshape(-1))


def softmax_probabilities(query, key, temperature=0.5):
    query = normalize(query)
    key = normalize(key)
    logits = query @ key.T
    logits = logits / temperature
    shifted_cov = logits - jax.lax.stop_gradient(logits.max(axis=-1, keepdims=True))  # [N, N]
    diag_indexes = jnp.arange(shifted_cov.shape[0])[:, None]  # [N, 1]
    pos = jnp.take_along_axis(arr=shifted_cov, indices=diag_indexes, axis=-1)  # [N, 1]
    pos = jnp.exp(pos)
    neg = jnp.sum(jnp.exp(logits), axis=-1, keepdims=True) # [N, 1]
    return pos / neg


def cpc_loss(
        query,
        key,
        temperature = 0.5
):

    query = normalize(query)
    key = normalize(key)
    cov = query @ key.T  # (N, N) positive pairs on the diagonal
    sim = jnp.exp(cov / temperature)
    neg = sim.sum(axis=-1) # b
    row_sub = jnp.ones_like(neg) * math.exp(1/temperature)
    neg = jnp.clip(neg - row_sub, a_min=1e-6)

    pos = jnp.exp(jnp.sum(query * key, axis=-1) / temperature) # b
    loss = -jnp.log(pos / (neg + 1e-6))
    return loss.mean()

if __name__ == "__main__":
    # x = jax.random.normal(key=jax.random.PRNGKey(5), shape=(15, 5))
    # 10, 5
    jax_input = jnp.array([[ 0.61735314,  0.65116936,  0.37252188,  0.01196358,
              -1.0840642 ],
             [ 0.40633643, -0.3350711 ,  0.433196  ,  1.8324155 ,
               1.2233032 ],
             [ 0.6076932 ,  0.62271905, -0.5155139 , -0.8686952 ,
               1.3694043 ],
             [ 1.5686233 , -1.0647503 ,  1.0048455 ,  1.4000669 ,
               0.30719075],
             [ 1.6678249 , -0.5851507 , -1.420454  , -0.05948697,
              -1.5111905 ],
             [ 1.8621138 , -0.6911869 , -0.94851583,  1.159258  ,
               1.5931036 ],
             [ 1.9720763 , -1.0973446 ,  1.1731594 ,  0.0780869 ,
               0.143219  ],
             [-1.0157285 ,  0.50870734,  0.39398482,  1.1644812 ,
              -0.26890013],
             [ 1.6161795 ,  1.644653  , -1.0968473 ,  1.0495588 ,
               0.47088355],
             [-0.13400784,  0.5755616 ,  0.4617284 ,  0.08174139,
              -1.0918598 ]])

    torch_input = torch.tensor([[ 0.61735314,  0.65116936,  0.37252188,  0.01196358,
              -1.0840642 ],
             [ 0.40633643, -0.3350711 ,  0.433196  ,  1.8324155 ,
               1.2233032 ],
             [ 0.6076932 ,  0.62271905, -0.5155139 , -0.8686952 ,
               1.3694043 ],
             [ 1.5686233 , -1.0647503 ,  1.0048455 ,  1.4000669 ,
               0.30719075],
             [ 1.6678249 , -0.5851507 , -1.420454  , -0.05948697,
              -1.5111905 ],
             [ 1.8621138 , -0.6911869 , -0.94851583,  1.159258  ,
               1.5931036 ],
             [ 1.9720763 , -1.0973446 ,  1.1731594 ,  0.0780869 ,
               0.143219  ],
             [-1.0157285 ,  0.50870734,  0.39398482,  1.1644812 ,
              -0.26890013],
             [ 1.6161795 ,  1.644653  , -1.0968473 ,  1.0495588 ,
               0.47088355],
             [-0.13400784,  0.5755616 ,  0.4617284 ,  0.08174139,
              -1.0918598 ]])

    ## TEST particle
    # knn_k = 3
    # knn_clip = 0.0
    # mean = 0.0
    # knn_avg = True
    # knn_rm = True
    # particle_based_entropy = partial(particle_based_entropy, knn_k=knn_k, knn_clip=knn_clip, knn_rm=knn_rm,
    #                                         knn_avg=knn_avg)
    # value = particle_based_entropy(rep=jax_input, mean=mean, step=1)
    # print(value)
    # rms = RMS('cpu')
    # pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rm, 'cpu')
    # value_torch = pbe(torch_input)
    # print(value_torch)

    ## TEST nce
    # out = noise_contrastive_loss(jax_input, jax_input)
    # out = cpc_loss(jax_input, jax_input)
    # print(out)
    # out_torch = torch_nce(torch_input, torch_input)
    # print(out_torch)
    # print("Sanity Check value should be close to log(1/N): {}".format(math.log(jax_input.shape[0])))