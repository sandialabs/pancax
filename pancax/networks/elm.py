from typing import Callable
import equinox as eqx
import jax


def activation(x):
    alpha = 3.
    return 1. / (1. + jax.numpy.exp(-alpha * x))


class ELM(eqx.Module):
  layer: any
  beta: any
  n_outputs: int

  def __init__(self, n_inputs, n_outputs, n_neurons, key):
    radius = 3.
    # setup linear layer to have normal distribtuion
    layer = eqx.nn.Linear(n_inputs, n_neurons, use_bias=True, key=key)
    k1, k2 = jax.random.split(key, 2)
    new_weight = radius * jax.random.normal(k1, (n_neurons, n_inputs))
    new_bias = radius * jax.random.normal(k2, (n_neurons,))
    layer = eqx.tree_at(lambda m: m.weight, layer, new_weight)
    layer = eqx.tree_at(lambda m: m.bias, layer, new_bias)
    self.layer = layer
    self.beta = jax.numpy.zeros((n_outputs * n_neurons,))
    self.n_outputs = n_outputs

  def __call__(self, x):
    n_neurons = self.layer.out_features
    beta_temp = self.beta.reshape((n_neurons, self.n_outputs))
    H = activation(self.layer(x))
    return H @ beta_temp


class ELM2(eqx.Module):
  layer: any
  beta: any
  n_outputs: int

  def __call__(self, x):
    n_neurons = self.layer.out_features
    beta_temp = self.beta.reshape((n_neurons, self.n_outputs))
    H = activation(self.layer(x))
    return H @ beta_temp
