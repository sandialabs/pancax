# from .fields import Field
from .mlp import MLP
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class MLDirichletField(eqx.Module):
    ensure_positivity: bool
    bcs_network: eqx.Module = eqx.field(static=True)
    sdf_network: eqx.Module = eqx.field(static=True)
    mlp_network: eqx.Module

    def __init__(
        self,
        problem,
        key,
        ensure_positivity: Optional[bool] = False
    ):
        n_dims = problem.n_dims
        n_dofs = problem.n_dofs

        sdf_network = MLP(n_dims, 1, 50, 3, jax.nn.tanh, key)
        bcs_network = MLP(n_dims + 1, n_dofs, 50, 3, jax.nn.tanh, key)
        mlp_network = MLP(n_dims + 1, n_dofs, 50, 3, jax.nn.tanh, key)
        # mlp_network = Field(problem, key)

        coords = problem.domain.mesh.coords

        # setup inputs
        x_true = []
        for bc in problem.dirichlet_bcs:
            x_true.append(bc.coordinates(problem.domain.mesh))

        x_true = jnp.vstack(x_true)
        x_true = (x_true - jnp.min(coords, axis=0)) / (
            jnp.max(coords, axis=0) - jnp.min(coords, axis=0)
        )
        y_true = jnp.zeros((x_true.shape[0], 1))

        # create simple mse loss
        @eqx.filter_value_and_grad
        def loss_func(params, x, y):
            # x = (x - jnp.min(x, axis=0)) /
            # (jnp.max(x, axis=0) - jnp.min(x, axis=0))
            y_pred = eqx.filter_vmap(params)(x)
            return jnp.square(y_pred - y).mean()

        @eqx.filter_jit
        def step(params, x, y, opt_st):
            loss, grad = loss_func(params, x, y)
            updates, opt_st = opt.update(grad, opt_st)
            params = eqx.apply_updates(params, updates)
            return loss, params, opt_st

        # create an optimizer and make its state
        opt = optax.adam(1e-3)
        opt_st = opt.init(eqx.filter(sdf_network, eqx.is_array))

        # train the sdf network
        # TODO add tolerance
        for n in range(1000):
            loss, sdf_network, opt_st = \
                step(sdf_network, x_true, y_true, opt_st)
            if n % 100 == 0:
                print(loss)

        # now create the bc map
        # setup inputs
        x_true = []
        y_true = []
        n_dofs = []
        for bc in problem.dirichlet_bcs:
            coords = bc.coordinates(problem.domain.mesh)
            for t in problem.times:
                temp = jnp.hstack((coords, t * jnp.ones((coords.shape[0], 1))))
                x_true.append(temp)
                y_true.append(
                    jax.vmap(lambda x: jnp.array(bc.function(x, t)))(
                        coords
                    ).reshape((-1, 1))
                )
                n_dofs.append(
                    bc.component * jnp.ones(
                        (coords.shape[0], 1),
                        dtype=jnp.int32
                    )
                )

        coords = problem.domain.mesh.coords
        mins = jnp.append(jnp.min(coords, axis=0), jnp.min(problem.times))
        maxs = jnp.append(jnp.max(coords, axis=0), 1.0)
        x_true = jnp.vstack(x_true)
        x_true = (x_true - mins) / (maxs - mins)
        y_true = jnp.vstack(y_true)
        n_dofs = jnp.vstack(n_dofs)

        # create simple mse loss
        @eqx.filter_value_and_grad
        def loss_func_2(params, x, y, d):
            # y_pred = eqx.filter_vmap(params)(x)[:, d]
            y_pred = eqx.filter_vmap(params)(x)
            return jnp.square(y_pred - y).mean()

        @eqx.filter_jit
        def step(params, x, y, opt_st):
            loss, grad = loss_func_2(params, x, y, n_dofs)
            updates, opt_st = opt.update(grad, opt_st)
            params = eqx.apply_updates(params, updates)
            return loss, params, opt_st

        # create an optimizer and make its state
        opt = optax.adam(1e-3)
        opt_st = opt.init(eqx.filter(bcs_network, eqx.is_array))

        # train the sdf network
        # TODO add tolerance
        for n in range(10000):
            loss, bcs_network, opt_st = \
                step(bcs_network, x_true, y_true, opt_st)
            if n % 100 == 0:
                print(loss)

        self.ensure_positivity = ensure_positivity
        self.sdf_network = sdf_network
        self.bcs_network = bcs_network
        self.mlp_network = mlp_network

    def __call__(self, x_in):
        x = x_in[:-1]
        # t = x_in[-1]
        z = self.mlp_network(x_in)
        z = self.sdf_network(x) * z
        u = self.bcs_network(x_in) + z
        # u = z

        # if self.ensure_positivity:
        #   u = jax.nn.softplus(u)

        return u
