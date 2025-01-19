from abc import abstractmethod
from jax import grad, hessian, jacfwd, value_and_grad, vmap
from jax.lax import stop_gradient
from jaxtyping import Array, Float
from pancax.fem import assemble_sparse_stiffness_matrix
from typing import Callable, Dict
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp


# TODO clean this up
def nodal_pp(func, has_props=False, jit=True):
	"""
	:param func: Function to use for a nodal property output variable
	:param has_props: Whether or not this function need properties
	:param jit: Whether or not to jit this function
	"""
	if has_props:
		# new_func = lambda p, d, t: vmap(
		# 	func, in_axes=(None, 0, None, None)
		# )(p.fields, d.coords, t, p.properties)
		new_func = lambda p, d, t: vmap(
			func, in_axes=(None, 0, None)
		)(p, d.coords, t)
	else:
		new_func = lambda p, d, t: vmap(
			func, in_axes=(None, 0, None)
		)(p.fields, d.coords, t)
	
	if jit:
		return eqx.filter_jit(new_func)
	else:
		return new_func


def standard_pp(physics):
	d = {
		'field_values': {
			'method': nodal_pp(physics.field_values),
			'names' : physics.field_value_names
		}
	}
	return d


class BasePhysics(eqx.Module):
  field_value_names: tuple[str, ...]  
  var_name_to_method: Dict[str, Callable] #= field(default_factory=lambda: {})
  # needs to be set
  # by default it just returns network output
  dirichlet_bc_func: Callable #= lambda x, t, z: z
  # TODO how to make this dimension dependent?
  # finally use generics in python?
  x_mins: Float[Array, "nd"] #= jnp.zeros(3)
  x_maxs: Float[Array, "nd"] #= jnp.zeros(3)

  def __init__(self, field_value_names: tuple[str, ...]) -> None:
    self.field_value_names = field_value_names
    self.var_name_to_method = {}
    self.dirichlet_bc_func = lambda x, t, z: z
    # TODO improve this error handling
    self.x_mins = jnp.zeros(3)
    self.x_maxs = jnp.zeros(3)

  # TODO need to modify for delta pinn
  def field_values(self, field, x, t, *args):
    x = (x - stop_gradient(self.x_mins)) / (stop_gradient(self.x_maxs) - stop_gradient(self.x_mins))
    inputs = jnp.hstack((x, t))
    z = field(inputs)
    u = self.dirichlet_bc_func(x, t, z)
    return u
  
  def field_gradients(self, field, x, t, *args):
    return jacfwd(self.field_values, argnums=1)(field, x, t, *args)

  def field_hessians(self, field, x, t, *args):
    return hessian(self.field_values, argnums=1)(field, x, t, *args)

  def field_laplacians(self, field, x, t, *args):
    return jnp.trace(self.field_hessians(field, x, t, *args), axis1=1, axis2=2)

  def field_time_derivatives(self, field, x, t, *args):
    return jacfwd(self.field_values, argnums=2)(field, x, t, *args)

  def field_second_time_derivatives(self, field, x, t, *args):
    return jacfwd(self.field_time_derivatives, argnums=2)(field, x, t, *args)

  def update_dirichlet_bc_func(self, bc_func: Callable):
    get_fn = lambda x: x.dirichlet_bc_func
    new_pytree = eqx.tree_at(get_fn, self, bc_func)
    return new_pytree

  def update_normalization(self, domain):
    x_mins = jnp.min(domain.coords, axis=0)
    x_maxs = jnp.max(domain.coords, axis=0)
    new_pytree = eqx.tree_at(lambda x: x.x_mins, self, x_mins)
    new_pytree = eqx.tree_at(lambda x: x.x_maxs, new_pytree, x_maxs)
    return new_pytree
  
  def update_var_name_to_method(self):
    var_name_to_method = standard_pp(self)
    new_pytree = eqx.tree_at(lambda x: x.var_name_to_method, self, var_name_to_method)
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
  pass


class BaseEnergyFormPhysics(BaseVariationalFormPhysics):
  field_value_names: tuple[str, ...]

  def element_energy(self, params, x, t, u, fspace, *args):
    vs = fspace.shape_function_values(x)
    grad_vs = fspace.shape_function_gradients(x)
    JxWs = fspace.JxWs(x)
    xs = vmap(lambda y: jnp.dot(y, x))(vs)
    us = vmap(lambda y: jnp.dot(y, u))(vs)
    grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
    in_axes = (None, 0, None, 0, 0) + len(args) * (None,)
    pis = vmap(self.energy, in_axes=in_axes)(params, xs, t, us, grad_us, *args)
    return jnp.dot(JxWs, pis)

  def element_kinetic_energy(self, params, x, t, u, fspace, *args):
    vs = fspace.shape_function_values(x)
    grad_vs = fspace.shape_function_gradients(x)
    JxWs = fspace.JxWs(x)
    xs = vmap(lambda y: jnp.dot(y, x))(vs)
    us = vmap(lambda y: jnp.dot(y, u))(vs)
    grad_us = vmap(lambda y: (y.T @ u).T)(grad_vs)
    in_axes = (None, 0, None, 0, 0) + len(args) * (None,)
    pis = vmap(self.kinetic_energy, in_axes=in_axes)(params, xs, t, us, grad_us, *args)
    return jnp.dot(JxWs, pis)

  @abstractmethod
  def energy(self, params, x, t, u, grad_u, *args):
    pass

  # def internal_force(self, params, domain, t, us, *args):
  #   return grad(self.potential_energy, argnums=3)(params, domain, t, us, *args)
  #   # def inner_func(params, domain, t):
       

  # TODO currently only valid for one block
  def potential_energy_on_block(self, params, x, t, us, fspace, conns, *args):
    # field, _ = params
    # us = self.field_values(field, x, t, *args)
    # us = self.vmap_field_values(params, x, t, *args)
    us = us[conns, :]
    xs = x[conns, :]
    return self.vmap_element_energy(params, xs, t, us, fspace, conns, *args)
  
  # TODO only works on a single block
  # def potential_energy(self, params, domain, t, *args):
  def potential_energy(self, params, domain, t, us, *args):
    return self.potential_energy_on_block(params, domain.coords, t, us, domain.fspace, domain.conns, *args)

  def potential_energy_and_internal_force(self, params, domain, t, us, *args):
    return value_and_grad(self.potential_energy, argnums=3)(params, domain, t, us, *args)

  def potential_energy_and_residual(self, params, domain, t, us, *args):
    pi, f = self.potential_energy_and_internal_force(params, domain, t, us, *args)
    return pi, jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])
  
  def potential_energy_residual_and_reaction_force(self, params, domain, t, us, *args):
    global_data = args[0]
    pi, f = self.potential_energy_and_internal_force(params, domain, t, us, *args)
    R = jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])
    reaction = jnp.sum(f[global_data.reaction_nodes, global_data.reaction_dof])
    return pi, R, reaction

  def mass_matrix(self, params, domain, t, us, *args):
    # us = self.vmap_field_values(params, domain.coords, t, *args)
    dof_manager = args[0]
    us = us[domain.conns, :]
    xs = domain.coords[domain.conns, :]
    func = jax.hessian(self.element_kinetic_energy, argnums=3)
    in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
    fs = vmap(func, in_axes=in_axes)(params, xs, t, us, domain.fspace, domain.conns, *args)
    return assemble_sparse_stiffness_matrix(
       onp.asarray(fs), onp.asarray(domain.conns), dof_manager
    )

  def stiffness_matrix(self, params, domain, t, us, *args):
    # us = self.vmap_field_values(params, domain.coords, t, *args)
    dof_manager = args[0]
    us = us[domain.conns, :]
    xs = domain.coords[domain.conns, :]
    func = jax.hessian(self.element_energy, argnums=3)
    in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
    fs = vmap(func, in_axes=in_axes)(params, xs, t, us, domain.fspace, domain.conns, *args)
    return assemble_sparse_stiffness_matrix(
       onp.asarray(fs), onp.asarray(domain.conns), dof_manager
    )

  def vmap_element_energy(self, params, x, t, u, fspace, conns, *args):
    in_axes = (None, 0, None, 0, None, 0) + len(args) * (None,)
    pis = vmap(self.element_energy, in_axes=in_axes)(params, x, t, u, fspace, conns, *args)
    return jnp.sum(pis)


class BaseStrongFormPhysics(BasePhysics):
  field_value_names: tuple[int, ...]

  def strong_form_neumann_bc(self, params, x, t, n, *args):
    assert False, 'Can\'t call this on the base method. Need to implement for physics.'

  @abstractmethod
  def strong_form_residual(self, params, x, t, *args):
    pass

  # def vmap_strong_form_residual(    # us = self.field_values(params, x, t, *args)
    # us = us[fspace.conns, :]
    # xs = x[fspace.conns, :])

# class BaseWeakFormPhysics(BasePhysics):
#   n_dofs: int
#   field_value_names: tuple[int, ...]

#   def element_field(self, params, x, t, func, u, fspace):
#     # x_el = x[conn, :]
#     # u_el = u[conn, :]
#     # return vmap(func, in_axes=(None, None, None, ))
#     v = fspace.shape_function_values(x)
#     grad_v = fspace.shape_function_gradients(x)
#     JxWs = fspace.JxWs(x)
#     return jnp
#     pass

#   def element_field_gradient(self, params, x, t, *args):
#     pass

#   def quadrature_field(self, params, x, t, *args):
#     pass
  
#   # def quadrature_field_gradient(self):
#   #   pass

#   @abstractmethod
#   def energy(self, params, x, t, u, grad_u, *args):
#     pass