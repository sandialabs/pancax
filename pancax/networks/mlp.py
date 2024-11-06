from .initialization import *
from typing import Callable
from typing import Optional
import equinox as eqx
import jax


# TODO should we convert these to actual classes?

def Linear(
  n_inputs: int,
  n_outputs: int,
  key: jax.random.PRNGKey
):
  """
  :param n_inputs: Number of inputs to linear layer
  :param n_outputs: Number of outputs of the linear layer
  :param key: rng key
  :return: Equinox Linear layer
  """
  model = eqx.nn.Linear(
    n_inputs, n_outputs,
    use_bias=False,
    key=key
  )
  model = eqx.tree_at(lambda l: l.weight, model, jnp.zeros((n_outputs, n_inputs), dtype=jnp.float64))
  return model


def MLP(
  n_inputs: int,
  n_outputs: int,
  n_neurons: int,
  n_layers: int,
  activation: Callable,
  key: jax.random.PRNGKey,
  use_final_bias: Optional[bool] = False,
  init_func: Optional[Callable] = trunc_init
):
  """
  :param n_inputs: Number of inputs to the MLP
  :param n_outputs: Number of outputs of the MLP
  :param n_neurons: Number of neurons in each hidden layer of the MLP
  :param n_layers: Number of hidden layers in the MLP
  :param activation: Activation function, e.g. tanh or relu
  :param key: rng key
  :param use_final_bias: Boolean for whether or not to use a bias
    vector in the final layer
  :return: Equinox MLP layer
  """
  model = eqx.nn.MLP(
    n_inputs, n_outputs, n_neurons, n_layers, 
    activation=activation, 
    use_final_bias=use_final_bias,
    key=key
  )
  # model = init_linear_weight(model, init_func, key)
  # model = init_linear(model, init_func, key)
  return model


def MLPBasis(
  n_inputs: int,
  n_neurons: int,
  n_layers: int,
  activation: Callable,
  key: jax.random.PRNGKey,
  init_func: Optional[Callable] = trunc_init
):
  return MLP(
    n_inputs, n_neurons, n_neurons, n_layers, 
    activation=activation, 
    use_final_bias=True,
    key=key,
    init_func=init_func
  )


# class MLP(BaseNetwork):
#   mlp: eqx.nn.MLP

#   def __init__(
#     self,
#     n_inputs: int,
#     n_outputs: int,
#     n_neurons: int,
#     n_layers: int,
#     activation: Callable,
#     key: jax.random.PRNGKey,
#     use_final_bias: Optional[bool] = False
#   ):
#     super().__init__(self)
#     model = eqx.nn.MLP(
#       n_inputs, n_outputs, n_neurons, n_layers, 
#       activation=activation, 
#       use_final_bias=use_final_bias,
#       key=key
#     )
#     model = init_linear_weight(model, trunc_init, key)
#     self.mlp = model

#   def __call__(self, params, x):
#     return self.mlp(params, x)
