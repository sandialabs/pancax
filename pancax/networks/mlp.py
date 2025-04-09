from .initialization import trunc_init
from typing import Callable
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp


# TODO should we convert these to actual classes?


def Linear(n_inputs: int, n_outputs: int, key: jax.random.PRNGKey):
    """
    :param n_inputs: Number of inputs to linear layer
    :param n_outputs: Number of outputs of the linear layer
    :param key: rng key
    :return: Equinox Linear layer
    """
    model = eqx.nn.Linear(n_inputs, n_outputs, use_bias=False, key=key)
    model = eqx.tree_at(
        lambda layer: layer.weight, model, jnp.zeros(
            (n_outputs, n_inputs),
            dtype=jnp.float64
        )
    )
    return model


def MLPBasis(
    n_inputs: int,
    n_neurons: int,
    n_layers: int,
    activation: Callable,
    key: jax.random.PRNGKey,
    init_func: Optional[Callable] = trunc_init,
):
    return MLP(
        n_inputs,
        n_neurons,
        n_neurons,
        n_layers,
        activation=activation,
        use_final_bias=True,
        key=key,
        init_func=init_func,
    )


class MLP(eqx.Module):
    mlp: eqx.Module

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_neurons: int,
        n_layers: int,
        activation: Callable,
        key: jax.random.PRNGKey,
        use_final_bias: Optional[bool] = False,
        init_func: Optional[Callable] = trunc_init,
        ensure_positivity: Optional[bool] = False,
    ):
        if ensure_positivity:
            def final_activation(x):
                return jax.nn.softplus(x)
        else:
            def final_activation(x):
                return x

        model = eqx.nn.MLP(
            n_inputs,
            n_outputs,
            n_neurons,
            n_layers,
            activation=activation,
            final_activation=final_activation,
            use_final_bias=use_final_bias,
            key=key,
        )
        self.mlp = model

    def __call__(self, x):
        return self.mlp(x)
