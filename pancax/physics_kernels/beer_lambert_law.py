from .base import BaseStrongFormPhysics
import jax.numpy as jnp


class BeerLambertLaw(BaseStrongFormPhysics):
  field_value_names: tuple[int, ...]
  
  def __init__(self):
    super().__init__(('I',))

  def strong_form_residual(self, params, x, t, *args):
    field, _ = params
