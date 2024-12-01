from .base_loss_function import BaseLossFunction
from jax import vmap
import jax.numpy as jnp

class ICLossFunction(BaseLossFunction):
  weight: float = 1.0

  def __call__(self, params, problem):
    def vmap_func(ic):
      field, _ = params
      coords = problem.domain.coords
      physics = problem.physics
      us_predicted = physics.vmap_field_values(field, coords, 0.)
      us_expected = vmap(ic)(coords)
      return jnp.square(us_predicted - us_expected).mean()
    
    error = 0.0
    for ic in problem.ics:
      error = error + vmap_func(ic)
    return self.weight * error, dict(ic=error)
