from .base import BaseStrongFormPhysics, BaseVariationalFormPhysics
from ..constitutive_models import Property
from jaxtyping import Array, Float
from typing import Union
import equinox as eqx
import jax.numpy as jnp


Vector = Union[Float[Array, "2"], Float[Array, "3"]]


class BeerLambertLaw(BaseStrongFormPhysics, BaseVariationalFormPhysics):
    field_value_names: tuple[int, ...]
    d: Vector = eqx.field(static=True)
    sigma: Property

    def __init__(self, d: Vector, sigma: Property):
        super().__init__(("I",))
        self.d = d
        self.sigma = sigma

    def residual(self, params, x, t, u, v, grad_u, grad_v, *args):
        # note this is not stabilized
        eq = jnp.dot(self.d, grad_u.T) + self.sigma * u
        return v * eq

    def strong_form_residual(self, params, x, t, *args):
        val_I = self.field_values(params, x, t, *args)
        grad_I = self.field_gradients(params, x, t, *args)
        return jnp.dot(self.d, grad_I.T) + self.sigma * val_I
