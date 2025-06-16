from typing import Callable, List
import equinox as eqx
import jax


class ResNetBlock(eqx.Module):
    activation: Callable
    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear
    shortcut: eqx.nn.Linear

    def __init__(
        self,
        n_neurons: int,
        activation: Callable,
        key: jax.random.PRNGKey
    ):
        key_1, key_2, key_3 = jax.random.split(key, 3)
        self.activation = activation
        self.linear_1 = eqx.nn.Linear(n_neurons, n_neurons, key=key_1)
        self.linear_2 = eqx.nn.Linear(n_neurons, n_neurons, key=key_2)
        self.shortcut = eqx.nn.Linear(n_neurons, n_neurons, key=key_3)

    def __call__(self, x):
        z = self.shortcut(x)
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        return self.activation(x + z)


class ResNet(eqx.Module):
    activation: Callable
    layers: List[ResNetBlock]

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_neurons: int,
        n_layers: int,
        activation: Callable,
        key: jax.random.PRNGKey
    ):
        layers = []
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(n_inputs, n_neurons, key=subkey))

        for n in range(n_layers):
            key, subkey = jax.random.split(key)
            layers.append(ResNetBlock(n_neurons, activation, key=subkey))

        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(n_neurons, n_outputs, key=subkey))

        self.activation = activation
        self.layers = layers

    def __call__(self, x):
        x = self.activation(self.layers[0](x))
        for n in range(1, len(self.layers) - 1):
            x = self.layers[n](x)
        x = self.layers[len(self.layers) - 1](x)
        return x
