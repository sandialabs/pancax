from .base import BaseStrongFormPhysics
import jax.numpy as jnp


class HeatEquation(BaseStrongFormPhysics):
  field_value_names: tuple[int]

  def __init__(self):
    super().__init__(('temperature'))

  def strong_form_residual(self, params, x, t, *args):
    field, props = params
    delta_u = self.field_laplacians(field, x, t, *args)
    dudt = self.field_time_derivatives(field, x, t, *args)
    rho, c, k = props
    return rho * c * dudt - k * delta_u
  
  def strong_form_neumann_bc(self, params, x, t, n, *args):
    field, props = params
    grad_u = self.field_gradients(field, x, t, *args)
    _, _, k = props
    return -k * jnp.dot(grad_u, n)
