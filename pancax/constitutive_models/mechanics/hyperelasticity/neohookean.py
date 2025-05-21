from ..base import HyperelasticModel, Scalar, Tensor
from ...properties import Property
import jax.numpy as jnp


class NeoHookean(HyperelasticModel):
    r"""
    NeoHookean model with the following model form

    $$
    \psi(\mathbf{F}) = \
    \frac{1}{2}K\left[\frac{1}{2}\left(J^2 - \ln J\right)\right] + \
    \frac{1}{2}G\left(\bar{I}_1 - 3\right)
    $$
    """

    bulk_modulus: Property
    shear_modulus: Property

    def energy(
        self,
        grad_u: Tensor, theta: Scalar, state_old, dt: Scalar
    ) -> Scalar:
        K, G = self.bulk_modulus, self.shear_modulus

        # kinematics
        J = self.jacobian(grad_u)
        I_1_bar = self.I1_bar(grad_u)

        # constitutive
        W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
        W_dev = 0.5 * G * (I_1_bar - 3.0)

        return W_vol + W_dev, state_old
