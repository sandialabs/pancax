from .base_kernel import StrongFormPhysicsKernel, WeakFormPhysicsKernel
from .base_kernel import nodal_pp, standard_pp, full_tensor_names_2D
from jax import jacfwd
from typing import Callable, Optional
import jax.numpy as jnp


class LinearElasticity2D(StrongFormPhysicsKernel, WeakFormPhysicsKernel):
  n_dofs = 2
  field_value_names = ['displ_x', 'displ_y']
  use_delta_pinn: bool

  def __init__(
    self, 
    mesh_file: str,
    bc_func: Callable,
    body_force: Callable = lambda x, t: jnp.zeros(2),
    use_delta_pinn: Optional[bool] = False
  ) -> None:
    super().__init__(mesh_file, bc_func, use_delta_pinn)
    self.body_force = body_force
    self.var_name_to_method = standard_pp(self)
    self.var_name_to_method['field_gradients'] = {
      'names': full_tensor_names_2D('grad_displ'),
      'method': nodal_pp(self.field_gradients)
    }
    self.var_name_to_method['linear_strain'] = {
      'names': full_tensor_names_2D('linear_strain'),
      'method': nodal_pp(self.linear_strain)
    }
    self.var_name_to_method['cauchy_stress'] = {
      'names': full_tensor_names_2D('cauchy_stress'),
      'method': nodal_pp(self.cauchy_stress, has_props=True)
    }

  def energy(self, x_el, u_el, N, grad_N, JxW, props):
    lambda_, mu = props()
    grad_u_q = (grad_N.T @ u_el).T
    strain = 0.5 * (grad_u_q + grad_u_q.T)
    return 0.5 * lambda_ * jnp.trace(strain)**2 + \
           mu * jnp.trace(strain @ strain)

  def strong_form_residual(self, params, x, t):
    div_stress = jacfwd(self.cauchy_stress, argnums=1)(params, x, t)
    div_stress = jnp.trace(div_stress, axis1=1, axis2=2)
    f = self.body_force(x, t)
    return div_stress + f

  def strong_form_neumann_bc(self, params, x, t, n):
    sigma = self.cauchy_stress(params, x, t)
    traction = sigma @ n
    return traction

  def linear_strain(self, field_network, x, t):
    grad_u = self.field_gradients(field_network, x, t)
    return 0.5 * (grad_u + grad_u.T)

  def cauchy_stress(self, params, x, t):
    field_network, props = params
    lambda_, mu = props()
    strain = self.linear_strain(field_network, x, t)
    stress = lambda_ * jnp.trace(strain) * jnp.eye(2) + \
             2. * mu * strain
    return stress
