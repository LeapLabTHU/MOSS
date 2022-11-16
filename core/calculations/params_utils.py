import haiku as hk
import jax
import tree

def count_param(params: hk.Params):
    params_count_list = [p.size for ((mod_name, x), p) in tree.flatten_with_path(params)]
    return sum(params_count_list)


def polyak_averaging(params: hk.Params,
                     target_params: hk.Params,
                     tau: float
):
    return jax.tree_multimap(
        lambda x, y: tau * x + (1 - tau) * y,
        params, target_params
    )
