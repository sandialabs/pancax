from .base import BaseStrongFormPhysics
import jax.numpy as jnp


class BurgersEquation(BaseStrongFormPhysics):
    field_value_names: tuple[int, ...]
    # v: float

    def __init__(self):
        super().__init__(("u", "v"))
        # self.v = v

    def strong_form_residual(self, params, x, t, *args):
        u = self.field_values(params, x, t, *args)
        grad_u = self.field_gradients(params, x, t, *args)
        dudt = self.field_time_derivatives(params, x, t, *args)
        # return dudt + 0.01 * jnp.dot(grad_u, grad_u.T)
        return dudt + jnp.dot(u, grad_u.T)
