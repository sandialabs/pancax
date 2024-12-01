from .base_loss_function import BCLossFunction
from jax import vmap
from typing import Optional
import jax.numpy as jnp


class DirichletBCLoss(BCLossFunction):
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    error = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    ).mean()
    return self.weight * error, dict(dirichlet_bc=error)

  def load_step(self, params, domain, t):
    field_network, props = params
    # TODO switch to using a jax.lax.scan below
    # error = 0.0
    # for bc in domain.essential_bcs:
    #   coords = domain.coords[domain.mesh.nodeSets[bc.nodeSet]]
    #   us_predicted = vmap(domain.physics.field_values, in_axes=(None, 0, None))(
    #     field_network, coords, t
    #   )[:, bc.component]
    #   us_expected = vmap(bc.function, in_axes=(0, None))(coords, t)
    #   error += jnp.square(us_predicted - us_expected).mean()
    
    # print(type(domain.essential_bcs))
    def vmap_func(bc):
      coords = domain.coords[domain.mesh.nodeSets[bc.nodeSet]]
      us_predicted = vmap(domain.physics.field_values, in_axes=(None, 0, None))(
        field_network, coords, t
      )[:, bc.component]
      us_expected = vmap(bc.function, in_axes=(0, None))(coords, t)
      return jnp.square(us_predicted - us_expected).mean()

    # errors = vmap(vmap_func)(domain.essential_bcs)
    # error = jnp.sum(errors)
    error = 0.0
    for bc in domain.essential_bcs:
      error = error + vmap_func(bc)
    return error


# NOTE below only supports zero neumann conditions currently
# NOTE this will break on delta PINNs maybe?
class NeumannBCLoss(BCLossFunction):
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    error = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    ).mean()
    return self.weight * error, dict(neumann_bc=error)

  def load_step(self, params, domain, t):
    func = domain.physics.strong_form_neumann_bc
    xs = domain.neumann_xs
    ns = domain.neumann_ns
    error = jnp.square(vmap(func, in_axes=(None, 0, None, 0))(
      params, xs, t, ns
    )).mean()
    return error
