from .base import AbstractPancaxModel
from jaxtyping import Array, Float
from .mlp import MLP
from typing import Callable, List, Optional, Union
import equinox as eqx
import jax
import jax.numpy as jnp


class Field(AbstractPancaxModel):
    dirichlet_bc_func: Callable
    networks: Union[eqx.Module, List[eqx.Module]]
    seperate_networks: bool
    t_min: Float[Array, "1"]
    t_max: Float[Array, "1"]
    x_mins: Float[Array, "nd"]  # = jnp.zeros(3)
    x_maxs: Float[Array, "nd"]  # = jnp.zeros(3)

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
        # initialize to behavior that doesn't use an analytic
        # dirichlet bc func
        dirichlet_bc_func: Optional[Callable] = lambda x, t, z: z,
        network_type: type = MLP
    ):
        n_dims = problem.n_dims
        n_dofs = problem.n_dofs

        self.dirichlet_bc_func = dirichlet_bc_func

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
        self.t_min = jnp.min(problem.times, axis=0)
        self.t_max = jnp.max(problem.times, axis=0)
        self.x_mins = jnp.min(problem.coords, axis=0)
        self.x_maxs = jnp.max(problem.coords, axis=0)

    # def __call__(self, x):
    def __call__(self, x, t):
        x_norm = (x - self.x_mins) / (self.x_maxs - self.x_mins)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        inputs = jnp.hstack((x_norm, t_norm))

        if self.seperate_networks:
            @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
            def func(params, x):
                return params(x)

            # z = func(self.networks, x)[:, 0]
            z = func(self.networks, inputs)[:, 0]
        else:
            # z = self.networks(x)
            z = self.networks(inputs)

        # TODO call dirichlet bc func

        return self.dirichlet_bc_func(x, t, z)
