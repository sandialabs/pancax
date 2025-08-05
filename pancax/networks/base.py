from jaxtyping import Float
from typing import Callable, Union
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random


def _apply_init(
    init_fn: Callable,
    *args,
    key: random.PRNGKey
):
    return init_fn(*args, key=key)

# TODO make this conform with new interface
# def box_init(layer: eqx.nn.Linear, key: jax.random.PRNGKey):
#     in_size = layer.in_features
#     out_size = layer.out_features
#     k1, k2 = jax.random.split(key, 2)
#     p = jax.random.uniform(k1, (out_size, in_size))
#     n = jax.random.normal(k2, (out_size, in_size))

#     # normalize normals
#     # for i in range(n.shape[0]):
#     #   n = n.at[i, :].set(n[i, :] / jnp.linalg.norm(n[i, :]))

#     # calculate p_max
#     # p_max = jnp.max(jnp.sign(n))
#     # p_max = jnp.max(jnp.array([0.0, p_max]))

#     # setup A and b one vector at a time
#     A = jnp.zeros((out_size, in_size))
#     b = jnp.zeros((out_size,))
#     for i in range(n.shape[0]):
#         p_temp = p[i, :]
#         n_temp = n[i, :] / jnp.linalg.norm(n[i, :])
#         p_max = jnp.max(jnp.array([0.0, jnp.max(jnp.sign(n_temp))]))
#         k = 1.0 / jnp.sum((p_max - p_temp) * n_temp)
#         A = A.at[i, :].set(k * n_temp)
#         b = b.at[i].set(k * jnp.dot(p_temp, n_temp))

#     # k = jnp.zeros((n.shape[0],))
#     # for i in range(n.shape[0]):
#     #   k = k.at[i].set(1. / jnp.sum((p_max - p[i, :]) * n[i, :]))

#     # A = jax.vmap(lambda k, n: k * n, in_axes=(0, 0))(k, n)
#     # b = k * jax.vmap(lambda x: jnp.sum(x), in_axes=(1,))(n @ p.T)
#     # print(A)
#     # print(b)
#     # assert False
#     return A, b


def trunc_init(params: Float, key: random.PRNGKey):
    stddev = jnp.sqrt(1 / params.shape[0])
    return stddev * jax.random.truncated_normal(
        key, shape=params.shape, lower=-2, upper=2
    )


def uniform_init(params: Float, key: random.PRNGKey):
    k = 1. / params.shape[0]
    return jax.random.uniform(
        key=key, shape=params.shape, minval=-k, maxval=k
    )


def zero_init(params: Float, key: random.PRNGKey):
    return jnp.zeros(params.shape)


class AbstractPancaxModel(eqx.Module):
    """
    Base class for pancax model parameters.

    This includes a few helper methods
    """

    def deserialise(self, f_name):
        self = eqx.tree_deserialise_leaves(f_name, self)
        return self

    def init(
        self,
        init_fn: Callable,
        filter_func: Union[None, Callable] = None,
        *,
        key: random.PRNGKey
    ):
        def get_leaves(m):
            return jax.tree_util.tree_leaves(m, is_leaf=filter_func)

        leaves = get_leaves(self)
        keys = random.split(key, len(leaves))
        new_leaves = []
        for key, leave in zip(keys, leaves):
            if hasattr(leave, "shape"):
                # case for arrays
                new_leaves.append(_apply_init(init_fn, leave, key=key))
            else:
                new_leaves.append(leave)

        return eqx.tree_at(get_leaves, self, new_leaves)

    def serialise(self, base_name, epoch):
        file_name = f"{base_name}_{str(epoch).zfill(7)}.eqx"
        print(f"Serialising current parameters to {file_name}")
        eqx.tree_serialise_leaves(file_name, self)
