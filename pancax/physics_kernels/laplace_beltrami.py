from .base import BaseEnergyFormPhysics
import jax.numpy as jnp


class LaplaceBeltrami(BaseEnergyFormPhysics):
    field_value_names: tuple[int, ...] = "u"

    def __init__(self):
        super().__init__(("u"))

    def energy(self, params, x, t, u, grad_u, state_old, dt, *args):
        return jnp.sum(0.5 * jnp.dot(grad_u, grad_u.T)), state_old

    def kinetic_energy(self, params, x, t, u, grad_u, state_old, dt, *args):
        return 0.5 * jnp.dot(u, u), state_old
