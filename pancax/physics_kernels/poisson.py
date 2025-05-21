from .base import BaseEnergyFormPhysics, BaseStrongFormPhysics
from typing import Callable
import jax.numpy as jnp


class Poisson(BaseEnergyFormPhysics, BaseStrongFormPhysics):
    field_value_names: tuple[str, ...] = "u"
    f: Callable

    def __init__(self, f: Callable) -> None:
        super().__init__(("u"))
        self.f = f

    def energy(self, params, x, t, u, grad_u, state_old, dt, *args):
        f = self.f(x)
        pi = 0.5 * jnp.dot(grad_u, grad_u.T) - f * u
        return jnp.sum(pi), state_old

    def strong_form_neumann_bc(self, params, x, t, n, *args):
        grad_u = self.field_gradients(params, x, t, *args)
        return -jnp.dot(grad_u, n)

    def strong_form_residual(self, params, x, t, *args):
        delta_u = self.field_laplacians(params, x, t, *args)
        f = self.f(x)
        return -delta_u - f
