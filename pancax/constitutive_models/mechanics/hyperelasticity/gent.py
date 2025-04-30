from ..base import HyperelasticModel
from ...properties import Property
import jax
import jax.numpy as jnp


class Gent(HyperelasticModel):
    r"""
    Gent model with the following model form

    $$
    \psi(\mathbf{F}) = \
    \frac{1}{2}K\left[\frac{1}{2}\left(J^2 - \ln J\right)\right] - \
    \frac{1}{2}GJ_m\ln\left(1 - \frac{\bar{I}_1 - 3}{J_m}\right)
    $$
    """

    bulk_modulus: Property
    shear_modulus: Property
    Jm_parameter: Property

    def energy(self, grad_u, theta, state_old, dt):
        # unpack properties
        K, G, Jm = self.bulk_modulus, self.shear_modulus, self.Jm_parameter

        # kinematics
        J = self.jacobian(grad_u)
        I_1_bar = self.I1_bar(grad_u)

        # guard rail
        check_value = I_1_bar > Jm + 3.0 - 0.001
        I_1_bar = jax.lax.cond(
            check_value, lambda x: Jm + 3.0 - 0.001, lambda x: x, I_1_bar
        )

        # constitutive
        W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
        W_dev = -0.5 * G * Jm * jnp.log(1.0 - ((I_1_bar - 3.0) / Jm))
        return W_vol + W_dev, state_old
