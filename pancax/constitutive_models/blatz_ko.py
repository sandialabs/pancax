from .base import BaseConstitutiveModel
from .properties import Property
import jax.numpy as jnp


class BlatzKo(BaseConstitutiveModel):
  shear_modulus: Property

  def energy(self, F):
    # unpack properties
    G = self.shear_modulus

    # kinematics
    I2 = self.I2(F)
    I3 = jnp.linalg.det(F.T @ F)

    # constitutive
    W = (G / 2.) * (I2 / I3 + 2 * jnp.sqrt(I3) - 5.)
    return W
