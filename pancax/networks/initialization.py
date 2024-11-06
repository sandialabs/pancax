from jaxtyping import Array, Float
from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp


def zero_init(key: jax.random.PRNGKey, shape) -> Float[Array, "no ni"]:
  """
  :param weight: current weight array for sizing
  :param key: rng key
  :return: A new set of weights
  """
  out, in_ = weight.shape
  return jnp.zeros(shape, dtype=jnp.float64)


def trunc_init(key: jax.random.PRNGKey, shape) -> Float[Array, "no ni"]:
  """
  :param weight: current weight array for sizing
  :param key: rng key
  :return: A new set of weights
  """
  stddev = jnp.sqrt(1 / shape[0])
  return stddev * jax.random.truncated_normal(key, shape=shape, lower=-2, upper=2)


def init_linear_weight(model: eqx.Module, init_fn: Callable, key: jax.random.PRNGKey) -> eqx.Module:
  """
  :param model: equinox  model
  :param init_fn: function to initialize weigth with
  :param key: rng key
  :return: a new equinox model
  """
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [
    x.weight
    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
    if is_linear(x)
  ]
  weights = get_weights(model)
  new_weights = [
    init_fn(subkey, weight.shape)
    for subkey, weight in zip(jax.random.split(key, len(weights)), weights)
  ]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  return new_model


def init_linear(model: eqx.Module, init_fn: Callable, key: jax.random.PRNGKey):
  """
  :param model: equinox  model
  :param init_fn: function to initialize weigth with
  :param key: rng key
  :return: a new equinox model
  """
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_biases = lambda m: [
    x.bias
    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
    if is_linear(x)
  ]
  get_weights = lambda m: [
    x.weight
    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
    if is_linear(x)
  ]
  get_layers = lambda m: [
    x
    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
    if is_linear(x)
  ]
  layers = get_layers(model)
  weights = get_weights(model)
  biases = get_biases(model)
  new_layers = [
    init_fn(layer, subkey)
    for layer, subkey in zip(layers, jax.random.split(key, len(layers)))
  ]
  new_weights = [x[0] for x in new_layers]
  new_biases = [x[1] for x in new_layers]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  new_model = eqx.tree_at(get_biases, model, new_biases)
  return new_model


def box_init(layer: eqx.nn.Linear, key: jax.random.PRNGKey):
  in_size = layer.in_features
  out_size = layer.out_features
  k1, k2 = jax.random.split(key, 2)
  p = jax.random.uniform(k1, (out_size, in_size))
  n = jax.random.normal(k2, (out_size, in_size))

  # normalize normals
  # for i in range(n.shape[0]):
  #   n = n.at[i, :].set(n[i, :] / jnp.linalg.norm(n[i, :]))

  # calculate p_max
  # p_max = jnp.max(jnp.sign(n))
  # p_max = jnp.max(jnp.array([0.0, p_max]))

  # setup A and b one vector at a time
  A = jnp.zeros((out_size, in_size))
  b = jnp.zeros((out_size,))
  for i in range(n.shape[0]):
    p_temp = p[i, :]
    n_temp = n[i, :] / jnp.linalg.norm(n[i, :])
    p_max = jnp.max(jnp.array([0.0, jnp.max(jnp.sign(n_temp))]))
    k = 1. / jnp.sum((p_max - p_temp) * n_temp)
    A = A.at[i, :].set(k * n_temp)
    b = b.at[i].set(k * jnp.dot(p_temp, n_temp))

  # k = jnp.zeros((n.shape[0],))
  # for i in range(n.shape[0]):
  #   k = k.at[i].set(1. / jnp.sum((p_max - p[i, :]) * n[i, :]))
  
  # A = jax.vmap(lambda k, n: k * n, in_axes=(0, 0))(k, n)
  # b = k * jax.vmap(lambda x: jnp.sum(x), in_axes=(1,))(n @ p.T)
  # print(A)
  # print(b)
  # assert False
  return A, b
