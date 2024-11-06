from jax import vmap
from pancax.physics import nodal_incompressibility_constraint
from typing import Optional
import jax.numpy as jnp


def StrongFormResidualLoss(weight: Optional[float] = 1.0):
  def loss_func(params, domain):
    times = domain.times
    func = domain.physics.strong_form_residual
    xs = domain.coords
    def vmap_func(params, xs, t):
      residuals = vmap(func, in_axes=(None, 0, None))(params, xs, t)
      residuals = residuals[domain.dof_manager.unknownIndices].flatten()
      residual = jnp.square(residuals).mean()
      return residual
    residual = vmap(vmap_func, in_axes=(None, None, 0))(params, xs, times)
    residual = residual.mean()
    return weight * residual, dict(residual=residual)
  return loss_func


# def StrongFormDirichletBCLoss(weight: Optional[float] = 1.0):
#   def loss_func(params, domain):
#     times = domain.times
#     func = domain.physics.field_values


# TODO need to add support for inhomogeneous neumann conditions
def StrongFormNeumannBCLoss(weight: Optional[float] = 1.0):
  def loss_func(params, domain):
    times = domain.times
    func = domain.physics.strong_form_neumann_bc
    xs = domain.neumann_inputs
    ns = domain.neumann_normals
    def vmap_func(params, xs, t, ns):
      residuals = vmap(func, in_axes=(None, 0, None, 0))(params, xs, t, ns)
      residual = jnp.square(residuals - domain.neumann_outputs).mean()
      return residual
    residual = vmap(vmap_func, in_axes=(None, None, 0, None))(params, xs, times, ns)
    residual = residual.mean()
    return weight * residual, dict(neumann=residual)
  return loss_func


def StrongFormIncompressibilityConstraint(weight: Optional[bool] = 1.0):
  def loss_func(params, domain):
    def vmap_func(params, domain, time):
      return nodal_incompressibility_constraint(params, domain, time)
    loss = jnp.mean(vmap(vmap_func, in_axes=(None, None, 0))(params, domain, domain.times))
    return weight * loss, dict(incompressibility_error=loss)
  return loss_func
