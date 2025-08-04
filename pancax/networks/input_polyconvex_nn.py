from typing import Callable, List, Optional
import equinox as eqx
import jax
import jax.numpy as jnp


# TODO still need to apply weight enforcement manually on weights
# of layers with pos in the name


def _icnn_init(key, shape):
    in_features = shape[0]
    k = 1. / in_features
    return jax.random.uniform(key=key, shape=shape, minval=-k, maxval=k)


class InputPolyconvexNN(eqx.Module):
    n_convex: int
    n_inputs: int
    n_layers: int
    x1_xx_pos: eqx.nn.Linear
    x1_xy: eqx.nn.Linear
    y1: eqx.nn.Linear
    x_h_plus_1_xx_pos: List[eqx.nn.Linear]
    x_h_plus_1_xy: List[eqx.nn.Linear]
    y_h_plus_1: List[eqx.nn.Linear]
    x_n_xx_pos: eqx.nn.Linear
    x_n_xy: eqx.nn.Linear
    activation_x: Callable
    activation_y: Callable

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_convex: int,
        activation_x: Callable,
        activation_y: Callable,
        # key: jax.random.PRNGKey,
        n_layers: Optional[int] = 3,
        n_neurons_x: Optional[int] = 40,
        n_neurons_y: Optional[int] = 30,
        *,
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, 3 + 3 * n_layers + 2)

        x1_xx_pos = eqx.nn.Linear(
            n_convex, n_neurons_x,
            use_bias=True,
            key=keys[0]
        )
        x1_xx_pos = eqx.tree_at(
            lambda x: x.weight,
            x1_xx_pos,
            _icnn_init(key=keys[0], shape=x1_xx_pos.weight.shape)
        )
        x1_xx_pos = eqx.tree_at(
            lambda x: x.bias,
            x1_xx_pos,
            _icnn_init(key=keys[0], shape=x1_xx_pos.bias.shape)
        )
        x1_xy = eqx.nn.Linear(
            n_neurons_y, n_neurons_x,
            use_bias=False,
            key=keys[1]
        )
        x1_xy = eqx.tree_at(
            lambda x: x.weight,
            x1_xy,
            _icnn_init(key=keys[1], shape=x1_xy.weight.shape)
        )
        y1 = eqx.nn.Linear(
            n_inputs - n_convex, n_neurons_y,
            use_bias=True,
            key=keys[2]
        )
        y1 = eqx.tree_at(
            lambda x: x.weight,
            y1,
            _icnn_init(key=keys[2], shape=y1.weight.shape)
        )
        y1 = eqx.tree_at(
            lambda x: x.bias,
            y1,
            _icnn_init(key=keys[2], shape=y1.bias.shape)
        )

        x_h_plus_1_xx_pos = []
        x_h_plus_1_xy = []
        y_h_plus_1 = []

        for n in range(n_layers):
            x_h_plus_1_xx_pos.append(eqx.nn.Linear(
                n_neurons_x, n_neurons_x,
                use_bias=True,
                key=keys[3 + 3 * n]
            ))
            x_h_plus_1_xx_pos[n] = eqx.tree_at(
                lambda x: x.weight,
                x_h_plus_1_xx_pos[n],
                _icnn_init(
                    key=keys[3 + 3 * n],
                    shape=x_h_plus_1_xx_pos[n].weight.shape
                )
            )
            x_h_plus_1_xx_pos[n] = eqx.tree_at(
                lambda x: x.bias,
                x_h_plus_1_xx_pos[n],
                _icnn_init(
                    key=keys[3 + 3 * n],
                    shape=x_h_plus_1_xx_pos[n].bias.shape
                )
            )
            x_h_plus_1_xy.append(eqx.nn.Linear(
                n_neurons_y, n_neurons_x,
                use_bias=False,
                key=keys[3 + 3 * n + 1]
            ))
            x_h_plus_1_xy[n] = eqx.tree_at(
                lambda x: x.weight,
                x_h_plus_1_xy[n],
                _icnn_init(
                    key=keys[3 + 3 * n + 1],
                    shape=x_h_plus_1_xy[n].weight.shape
                )
            )
            y_h_plus_1.append(eqx.nn.Linear(
                n_neurons_y, n_neurons_y,
                use_bias=True,
                key=keys[3 + 3 * n + 2]
            ))
            y_h_plus_1[n] = eqx.tree_at(
                lambda x: x.weight,
                y_h_plus_1[n],
                _icnn_init(
                    key=keys[3 + 3 * n + 2],
                    shape=y_h_plus_1[n].weight.shape
                )
            )
            y_h_plus_1[n] = eqx.tree_at(
                lambda x: x.bias,
                y_h_plus_1[n],
                _icnn_init(
                    key=keys[3 + 3 * n + 2],
                    shape=y_h_plus_1[n].bias.shape
                )
            )

        x_n_xx_pos = eqx.nn.Linear(
            n_neurons_x, n_outputs,
            use_bias=True,
            key=keys[3 * (n + 1)]
        )
        x_n_xx_pos = eqx.tree_at(
            lambda x: x.weight,
            x_n_xx_pos,
            _icnn_init(key=keys[3 * (n + 1)], shape=x_n_xx_pos.weight.shape)
        )
        x_n_xx_pos = eqx.tree_at(
            lambda x: x.bias,
            x_n_xx_pos,
            _icnn_init(key=keys[3 * (n + 1)], shape=x_n_xx_pos.bias.shape)
        )
        x_n_xy = eqx.nn.Linear(
            n_neurons_y, n_outputs,
            use_bias=False,
            key=keys[3 * (n + 1) + 1]
        )
        x_n_xy = eqx.tree_at(
            lambda x: x.weight,
            x_n_xy,
            _icnn_init(key=keys[3 * (n + 1) + 1], shape=x_n_xy.weight.shape)
        )

        # finally set fields
        self.n_convex = n_convex
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.x1_xx_pos = x1_xx_pos
        self.x1_xy = x1_xy
        self.y1 = y1
        self.x_h_plus_1_xx_pos = x_h_plus_1_xx_pos
        self.x_h_plus_1_xy = x_h_plus_1_xy
        self.y_h_plus_1 = y_h_plus_1
        self.x_n_xx_pos = x_n_xx_pos
        self.x_n_xy = x_n_xy
        self.activation_x = activation_x
        self.activation_y = activation_y

    def __call__(self, x_in):
        n_nonconvex = self.n_inputs - self.n_convex
        y0 = x_in[0:n_nonconvex]
        y = self.y1(y0)
        x0 = x_in[n_nonconvex:]
        x = self.x1_xx_pos(x0) + self.x1_xy(y)
        x = self.activation_x(x)

        for layer in range(self.n_layers):
            y = self.y_h_plus_1[layer](y)
            y = self.activation_y(y)
            x = self.x_h_plus_1_xx_pos[layer](x) + self.x_h_plus_1_xy[layer](y)
            x = self.activation_x(x)

        z = self.x_n_xx_pos(x) + self.x_n_xy(y)
        return z

    def parameter_enforcement(self):
        temp = jnp.clip(self.x1_xx_pos.weight, min=1e-3)
        self = eqx.tree_at(lambda x: x.x1_xx_pos.weight, self, temp)

        for n in range(self.n_layers):
            temp = jnp.clip(self.x_h_plus_1_xx_pos[n].weight, min=1e-3)
            self = eqx.tree_at(
                lambda x: x.x_h_plus_1_xx_pos[n].weight, self, temp
            )

        temp = jnp.clip(self.x_n_xx_pos.weight, min=1e-3)
        self = eqx.tree_at(lambda x: x.x_n_xx_pos.weight, self, temp)
        return self
