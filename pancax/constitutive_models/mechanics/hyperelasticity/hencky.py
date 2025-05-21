from ..base import HyperelasticModel, Scalar, Tensor
from ...properties import Property
import jax.numpy as jnp


class Hencky(HyperelasticModel):
    bulk_modulus: Property
    shear_modulus: Property

    def energy(self, grad_u: Tensor, theta, state_old, dt) -> Scalar:
        K, G = self.bulk_modulus, self.shear_modulus

        # kinematics
        E = self.log_strain(grad_u)
        trE = jnp.trace(E)
        psi = 0.5 * K * trE**2 + G * jnp.tensordot(E, E)
        # G * tensor_math.norm_of_deviator_squared(E)
        return psi, state_old
