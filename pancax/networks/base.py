from jaxtyping import Float
from typing import Callable, Union
import equinox as eqx
import jax
import jax.random as random


def _apply_init(init_fn: Callable, params: Float, key: random.PRNGKey):
    return init_fn(params, key)


def uniform_init(params: Float, key: random.PRNGKey):
    k = 1. / params.shape[0]
    return jax.random.uniform(
        key=key, shape=params.shape, minval=-k, maxval=k
    )


class BasePancaxModel(eqx.Module):
    """
    Base class for pancax model parameters.

    This includes a few helper methods
    """

    def init(
        self,
        init_fn: Callable,
        filter_func: Union[None, Callable] = None,
        *,
        key: random.PRNGKey
    ):
        def get_leaves(m):
            return jax.tree_util.tree_laves(m, is_leaf=filter_func)

        leaves = get_leaves(self)
        keys = random.split(key, len(leaves))
        new_leaves = []
        for key, leave in zip(keys, leaves):
            new_leaves.append(_apply_init(init_fn, leave, key))

        return eqx.tree_at(get_leaves, self, new_leaves)

    def serialise(self, base_name, epoch):
        file_name = f"{base_name}_{str(epoch).zfill(7)}.eqx"
        print(f"Serialising current parameters to {file_name}")
        eqx.tree_serialise_leaves(file_name, self)
