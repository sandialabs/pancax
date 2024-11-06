from .base_kernel import WeakFormPhysicsKernel
from .base_kernel import nodal_pp
from typing import Optional
import jax.numpy as jnp


class LaplaceBeltrami(WeakFormPhysicsKernel):
  n_dofs = 1
  field_names = ['eigenvector']
  use_delta_pinn: bool

  def __init__(self, mesh_file, bc_func, n_eigs: int, use_delta_pinn: Optional[bool] = False):
    super().__init__(mesh_file, bc_func, use_delta_pinn)
    self.n_eigs = n_eigs
    self.var_name_to_method = {
      'eigenvector': {
        'method': nodal_pp(self.field_values),
        'names' : self.field_names
      }
    }

  def energy(self, x_el, u_el, N, grad_N, JxW, props):
    # caclulate quadrature level fields
    grad_u_q = (grad_N.T @ u_el).T

    # kernel specific stuff here
    pi_q = 0.5 * jnp.dot(grad_u_q, grad_u_q.T)

    return JxW * pi_q

  def kinetic_energy(self, x_el, u_el, N, grad_N, JxW, props):
    return JxW * 0.5 * jnp.dot(u_el.T, u_el)
