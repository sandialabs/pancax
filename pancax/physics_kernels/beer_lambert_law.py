from .base import BaseStrongFormPhysics
from ..constitutive_models import Property
from jaxtyping import Array, Float
import equinox as eqx


class BeerLambertLaw(BaseStrongFormPhysics):
  field_value_names: tuple[int, ...]
  d: Float[Array, "3"] = eqx.field(static=True)
  sigma: Property
  
  def __init__(self, d: Float[Array, "3"], sigma: Property):
    super().__init__(('I',))
    self.d = d
    self.sigma = sigma

  def strong_form_residual(self, params, x, t, *args):
    I = self.field_values(params, x, t, *args)
    grad_I = self.field_gradients(params, x, t, *args)
    return self.d @ grad_I + self.sigma * I
