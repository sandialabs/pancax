from abc import abstractmethod
from jax import hessian, jacfwd, value_and_grad, vmap
from jaxtyping import Array, Float
from pancax.fem import assemble_sparse_stiffness_matrix
from typing import Callable, Dict
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp


def element_pp(
    func,
    physics,
    is_kinematic_method=False,
    is_state_method=False,
    jit=True
):
    def kinematic_method(func, params, domain, t, us, state_old, dt, *args):
        coords, conns, fspace = domain.coords, domain.conns, domain.fspace
        us = us[conns, :]
        xs = coords[conns, :]

        def _vmap_func(x, u):
            vs = fspace.shape_function_values(x)
            grad_vs = fspace.shape_function_gradients(x)
            JxWs = fspace.JxWs(x)
            xs = vmap(lambda y: jnp.dot(y, x))(vs)
            us = vmap(lambda y: jnp.dot(y, u))(vs)
            grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
            return xs, us, grad_us, JxWs

        xs, us, grad_us, JxWs = vmap(_vmap_func, in_axes=(0, 0))(xs, us)
        grad_us = vmap(vmap(physics.formulation.modify_field_gradient))(
            grad_us
        )

        vals = vmap(vmap(func))(grad_us)
        return vals

    def state_method(func, params, domain, t, us, state_old, dt, *args):
        coords, conns, fspace = domain.coords, domain.conns, domain.fspace
        us = us[conns, :]
        xs = coords[conns, :]

        def _vmap_func(x, u):
            vs = fspace.shape_function_values(x)
            grad_vs = fspace.shape_function_gradients(x)
            JxWs = fspace.JxWs(x)
            xs = vmap(lambda y: jnp.dot(y, x))(vs)
            us = vmap(lambda y: jnp.dot(y, u))(vs)
            grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
            return xs, us, grad_us, JxWs

        xs, us, grad_us, JxWs = vmap(_vmap_func, in_axes=(0, 0))(xs, us)
        in_axes_1 = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)
        in_axes_2 = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)

        _, state_news = vmap(
            vmap(func, in_axes=in_axes_2), in_axes=in_axes_1)(
                params, xs, t, us, grad_us, state_old, dt
        )
        return state_news
        # vals = vmap(vmap(func))(grad_us)
        # return vals

    if is_kinematic_method:
        def new_func(p, d, t, u, s, dt, *args):
            return kinematic_method(
                # physics.constitutive_model.deformation_gradient,
                func,
                p, d, t, u, s, dt, *args
            )
    elif is_state_method:
        def new_func(p, d, t, u, s, dt, *args):
            return state_method(
                physics.energy,
                p, d, t, u, s, dt, *args
            )
    else:
        assert False, 'Only kinematic methods are currently supported'

    if jit:
        new_func = eqx.filter_jit(new_func)

    return new_func


# TODO clean this up
def nodal_pp(
    func,
    # has_props=False,
    physics,
    is_kinematic_method=False,
    jit=True
):
    """
    :param func: Function to use for a nodal property output variable
    :param has_props: Whether or not this function need properties
    :param jit: Whether or not to jit this function
    """
    if is_kinematic_method:
        def new_func(p, d, t):
            return vmap(func, in_axes=(None, 0, None))(p.fields, d.coords, t)
    else:
        def new_func(p, d, t, u, s, dt, *args):
            return func(p, d, t, u, s, dt, *args)

    if jit:
        return eqx.filter_jit(new_func)
    else:
        return new_func


def standard_pp(physics):
    d = {
        "field_values": {
            "method": nodal_pp(
                physics.field_values, physics,
                is_kinematic_method=True
            ),
            "names": physics.field_value_names,
        }
    }
    return d


class BasePhysics(eqx.Module):
    field_value_names: tuple[str, ...]
    var_name_to_method: Dict[str, Callable]
    # needs to be set
    # by default it just returns network output
    dirichlet_bc_func: Callable  # = lambda x, t, z: z
    # TODO how to make this dimension dependent?
    # finally use generics in python?
    # maybe loop in the mechanics formulation here and rename it?
    x_mins: Float[Array, "nd"]  # = jnp.zeros(3)
    x_maxs: Float[Array, "nd"]  # = jnp.zeros(3)

    def __init__(self, field_value_names: tuple[str, ...]) -> None:
        self.field_value_names = field_value_names
        self.var_name_to_method = {}
        self.dirichlet_bc_func = lambda x, t, z: z
        # TODO improve this error handling
        self.x_mins = jnp.zeros(3)
        self.x_maxs = jnp.zeros(3)

    # TODO need to modify for delta pinn
    def field_values(self, field, x, t, *args):
        # x = (x - stop_gradient(self.x_mins)) /
        #   (stop_gradient(self.x_maxs) - stop_gradient(self.x_mins))
        x = (x - self.x_mins) / (self.x_maxs - self.x_mins)
        inputs = jnp.hstack((x, t))
        z = field(inputs)
        u = self.dirichlet_bc_func(x, t, z)
        return u

    def field_gradients(self, field, x, t, *args):
        return jacfwd(self.field_values, argnums=1)(field, x, t, *args)

    def field_hessians(self, field, x, t, *args):
        return hessian(self.field_values, argnums=1)(field, x, t, *args)

    def field_laplacians(self, field, x, t, *args):
        return jnp.trace(
            self.field_hessians(field, x, t, *args), axis1=1, axis2=2
        )

    def field_time_derivatives(self, field, x, t, *args):
        return jacfwd(self.field_values, argnums=2)(field, x, t, *args)

    def field_second_time_derivatives(self, field, x, t, *args):
        return jacfwd(self.field_time_derivatives, argnums=2)(
            field, x, t, *args
        )

    @property
    def num_state_variables(self):
        return 0

    def update_dirichlet_bc_func(self, bc_func: Callable):
        # get_fn = lambda x: x.dirichlet_bc_func
        def get_fn(x):
            return x.dirichlet_bc_func
        new_pytree = eqx.tree_at(get_fn, self, bc_func)
        return new_pytree

    def update_normalization(self, domain):
        x_mins = jnp.min(domain.coords, axis=0)
        x_maxs = jnp.max(domain.coords, axis=0)

        # x_mins = jnp.append(x_mins, jnp.min(domain.times))
        # x_maxs = jnp.append(x_maxs, jnp.max(domain.times))

        new_pytree = eqx.tree_at(lambda x: x.x_mins, self, x_mins)
        new_pytree = eqx.tree_at(lambda x: x.x_maxs, new_pytree, x_maxs)
        return new_pytree

    def update_var_name_to_method(self):
        var_name_to_method = standard_pp(self)
        new_pytree = eqx.tree_at(
            lambda x: x.var_name_to_method, self, var_name_to_method
        )
        return new_pytree

    # some helpful vmaps
    def vmap_field_values(self, field, xs, t, *args):
        in_axes = (None, 0, None) + len(args) * (None,)
        return vmap(self.field_values, in_axes=in_axes)(field, xs, t, *args)

    def vmap_field_gradients(self, field, xs, t, *args):
        in_axes = (None, 0, None) + len(args) * (None,)
        return vmap(self.field_gradients, in_axes=in_axes)(field, xs, t, *args)

    @property
    def n_dofs(self):
        return len(self.field_value_names)


class BaseVariationalFormPhysics(BasePhysics):
    field_value_names: tuple[str, ...]

    def element_residual(self, params, x, t, u, fspace, *args):
        vs = fspace.shape_function_values(x)
        grad_vs = fspace.shape_function_gradients(x)
        JxWs = fspace.JxWs(x)
        xs = vmap(lambda y: jnp.dot(y, x))(vs)
        us = vmap(lambda y: jnp.dot(y, u))(vs)
        grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
        in_axes = (None, 0, None, 0, 0, 0, 0) + len(args) * (None,)
        pis = vmap(self.residual, in_axes=in_axes)(
            params, xs, t, us, vs, grad_us, grad_vs, *args
        )
        return jnp.dot(JxWs, pis)

    @abstractmethod
    def residual(self, params, x, t, u, v, grad_u, grad_v, *args):
        pass

    def vmap_element_residual(self, params, domain, t, us, *args):
        conns, fspace = domain.conns, domain.fspace
        us = us[domain.conns, :]
        xs = domain.coords[domain.conns, :]

        in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
        rs = vmap(self.element_residual, in_axes=in_axes)(
            params, xs, t, us, fspace, conns, *args
        )
        return rs


class BaseEnergyFormPhysics(BasePhysics):
    field_value_names: tuple[str, ...]

    # only used for delta pinn LBFO generation
    # TODO figure out how to reconcile this
    def element_energy_old(self, params, x, t, u, fspace, *args):
        vs = fspace.shape_function_values(x)
        grad_vs = fspace.shape_function_gradients(x)
        JxWs = fspace.JxWs(x)
        xs = vmap(lambda y: jnp.dot(y, x))(vs)
        us = vmap(lambda y: jnp.dot(y, u))(vs)
        grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
        in_axes = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)

        # hack for now
        state = jnp.zeros((us.shape[0], 0))
        dt = 0.
        pis, _ = vmap(self.energy, in_axes=in_axes)(
            params, xs, t, us, grad_us, state, dt, *args
        )
        return jnp.dot(JxWs, pis)

    def element_energy(
        self,
        params, xs, t, us, grad_us, JxWs, state_old, dt, *args
    ):
        in_axes = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)
        pis, state_new = vmap(self.energy, in_axes=in_axes)(
            params, xs, t, us, grad_us, state_old, dt, *args
        )
        return jnp.dot(JxWs, pis), state_new

    # only used for delta pinn LBFO generation
    # TODO figure out how to reconcile this
    def element_kinetic_energy_old(self, params, x, t, u, fspace, *args):
        vs = fspace.shape_function_values(x)
        grad_vs = fspace.shape_function_gradients(x)
        JxWs = fspace.JxWs(x)
        xs = vmap(lambda y: jnp.dot(y, x))(vs)
        us = vmap(lambda y: jnp.dot(y, u))(vs)
        grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
        in_axes = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)
        state_old = jnp.zeros((us.shape[0], 0))
        dt = 0.
        pis, state_new = vmap(self.kinetic_energy, in_axes=in_axes)(
            params, xs, t, us, grad_us, state_old, dt, *args
        )
        return jnp.dot(JxWs, pis)

    def element_kinetic_energy(
        self,
        params, xs, t, us, grad_us, JxWs, state_old, dt, *args
    ):
        in_axes = (None, 0, None, 0, 0, 0, None) + len(args) * (None,)
        pis, state_new = vmap(self.kinetic_energy, in_axes=in_axes)(
            params, xs, t, us, grad_us, state_old, dt, *args
        )
        return jnp.dot(JxWs, pis), state_new

    @abstractmethod
    def energy(self, params, x, t, u, grad_u, *args):
        pass

    # def internal_force(self, params, domain, t, us, *args):
    #   return grad(self.potential_energy, argnums=3)(
    #       params, domain, t, us, *args
    #   )
    #   # def inner_func(params, domain, t):

    # TODO currently only valid for one block
    def potential_energy_on_block(
        self, params, x, t, us, fspace, conns, state_old, dt, *args
    ):
        us = us[conns, :]
        xs = x[conns, :]

        def _vmap_func(x, u):
            vs = fspace.shape_function_values(x)
            grad_vs = fspace.shape_function_gradients(x)
            JxWs = fspace.JxWs(x)
            xs = vmap(lambda y: jnp.dot(y, x))(vs)
            us = vmap(lambda y: jnp.dot(y, u))(vs)
            grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
            return xs, us, grad_us, JxWs

        xs, us, grad_us, JxWs = vmap(_vmap_func, in_axes=(0, 0))(xs, us)

        return self.vmap_element_energy(
            params, xs, t, us, grad_us, JxWs, state_old, dt, *args
        )

    # TODO only works on a single block
    def potential_energy(self, params, domain, t, us, state_old, dt, *args):
        pi, state_new = self.potential_energy_on_block(
            params, domain.coords, t,
            us, domain.fspace, domain.conns, state_old, dt, *args
        )
        return pi, state_new

    def potential_energy_and_internal_force(
        self, params, domain, t, us, state_old, dt, *args
    ):
        return value_and_grad(self.potential_energy, argnums=3, has_aux=True)(
            params, domain, t, us, state_old, dt, *args
        )

    def potential_energy_and_residual(
        self, params, domain, t, us, state_old, dt, *args
    ):
        (pi, state_new), f = self.potential_energy_and_internal_force(
            params, domain, t, us, state_old, dt, *args
        )
        return (pi, state_new), jnp.linalg.norm(
            f.flatten()[domain.dof_manager.unknownIndices]
        )

    def potential_energy_residual_and_reaction_force(
        self, params, domain, t, us, state_old, dt, *args
    ):
        global_data = args[0]
        pi, f = self.potential_energy_and_internal_force(
            params, domain, t, us, state_old, dt, *args
        )
        R = jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])
        reaction = jnp.sum(
            f[global_data.reaction_nodes, global_data.reaction_dof]
        )
        return pi, R, reaction

    def mass_matrix(self, params, domain, t, us, state_old, dt, *args):
        # us = self.vmap_field_values(params, domain.coords, t, *args)
        dof_manager = args[0]
        us = us[domain.conns, :]
        xs = domain.coords[domain.conns, :]
        func = jax.hessian(self.element_kinetic_energy_old, argnums=3)
        in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
        fs = vmap(func, in_axes=in_axes)(
            params, xs, t, us, domain.fspace, domain.conns, *args
        )
        return assemble_sparse_stiffness_matrix(
            onp.asarray(fs), onp.asarray(domain.conns), dof_manager
        )

    def stiffness_matrix(self, params, domain, t, us, state_old, dt, *args):
        # us = self.vmap_field_values(params, domain.coords, t, *args)
        dof_manager = args[0]
        us = us[domain.conns, :]
        xs = domain.coords[domain.conns, :]
        func = jax.hessian(self.element_energy_old, argnums=3)
        in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
        fs = vmap(func, in_axes=in_axes)(
            params, xs, t, us, domain.fspace, domain.conns, *args
        )
        return assemble_sparse_stiffness_matrix(
            onp.asarray(fs), onp.asarray(domain.conns), dof_manager
        )

    def vmap_element_energy(
        self,
        params, xs, t, us, grad_us, JxWs, state_old, dt, *args
    ):
        in_axes = (None, 0, None, 0, 0, 0, 0, None) + len(args) * (None,)
        pis, state_new = vmap(self.element_energy, in_axes=in_axes)(
            params, xs, t, us, grad_us, JxWs, state_old, dt, *args
        )
        return jnp.sum(pis), state_new


class BaseStrongFormPhysics(BasePhysics):
    field_value_names: tuple[int, ...]

    def strong_form_neumann_bc(self, params, x, t, n, *args):
        assert (
            False
        ), "Can't call this on the base method. Need to implement for physics."

    @abstractmethod
    def strong_form_residual(self, params, x, t, *args):
        pass
