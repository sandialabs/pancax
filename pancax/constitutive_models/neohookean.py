from .base import BaseConstitutiveModel, Scalar, Tensor
from .properties import Property
import jax.numpy as jnp


class NeoHookean(BaseConstitutiveModel):
  r"""
	NeoHookean model with the following model form

	$$
  \psi(\mathbf{F}) = \frac{1}{2}K\left[\frac{1}{2}\left(J^2 - \ln J\right)\right] + \frac{1}{2}G\left(\bar{I}_1 - 3\right)
  $$
	"""
  bulk_modulus: Property
  shear_modulus: Property

  def energy(self, F: Tensor) -> Scalar:
    K, G = self.bulk_modulus, self.shear_modulus

    # kinematics
    # F = jnp.eye(3) + grad_u
    C = F.T @ F
    J = self.jacobian(F)
    I_1_bar = jnp.trace(1. / jnp.square(jnp.cbrt(J)) * C)

    # constitutive
    W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
    W_dev = 0.5 * G * (I_1_bar - 3.)

    return W_vol + W_dev
