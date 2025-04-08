from ..base import HyperelasticModel
from ...properties import Property
import jax.numpy as jnp


class BlatzKo(HyperelasticModel):
    shear_modulus: Property

    def energy(self, grad_u):
        # unpack properties
        G = self.shear_modulus

        # kinematics
        I2 = self.I2(grad_u)
        I3 = self.I3(grad_u)

        # constitutive
        W = (G / 2.0) * (I2 / I3 + 2 * jnp.sqrt(I3) - 5.0)
        return W
