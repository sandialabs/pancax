from .base import BasePancaxModel
from .mlp import MLP
from typing import Callable, List, Optional, Union
import equinox as eqx
import jax


class Field(BasePancaxModel):
    networks: Union[eqx.Module, List[eqx.Module]]
    seperate_networks: bool

    def __init__(
        self,
        problem,
        key: jax.random.PRNGKey,
        ensure_positivity: Optional[bool] = False,
        seperate_networks: Optional[bool] = False,
        # specific options below
        # MLP
        n_layers: Optional[int] = 3,
        n_neurons: Optional[int] = 50,
        activation: Optional[Callable] = jax.nn.tanh,
        #
        network_type: type = MLP
    ):
        n_dims = problem.n_dims
        n_dofs = problem.n_dofs

        if seperate_networks:

            @eqx.filter_vmap
            def init(k):
                return network_type(
                    n_dims + 1,
                    1,
                    n_neurons,
                    n_layers,
                    activation,
                    k,
                    # ensure_positivity=ensure_positivity,
                )

            self.networks = init(jax.random.split(key, n_dofs))
        else:
            self.networks = network_type(
                n_dims + 1,
                n_dofs,
                n_neurons,
                n_layers,
                activation,
                key,
                # ensure_positivity=ensure_positivity,
            )

        self.seperate_networks = seperate_networks

    def __call__(self, x):
        if self.seperate_networks:

            @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
            def func(params, x):
                return params(x)

            return func(self.networks, x)[:, 0]
        else:
            return self.networks(x)
