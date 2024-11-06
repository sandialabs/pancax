from .base_loss_function import PhysicsLossFunction
from jax import vmap
from typing import Optional
import jax.numpy as jnp


# NOTE this probably does not currently dsupport deltaPINNs
class StrongFormResidualLoss(PhysicsLossFunction):
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    residual = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    ).mean()
    return self.weight * residual, dict(residual=residual)

  def load_step(self, params, domain, t):
    func = domain.physics.strong_form_residual
    # TODO this will fail on delta PINNs currently
    residuals = vmap(func, in_axes=(None, 0, None))(params, domain.coords, t)
    return jnp.square(residuals).mean()
