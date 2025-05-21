from ..base import HyperelasticModel
from ...properties import Property
import equinox as eqx
import jax.numpy as jnp


class Swanson(HyperelasticModel):
    r"""
    Swanson model truncated to 4 parameters

    $$
    \psi(\mathbf{F}) = K\left(J\ln J - J + 1\right) + \
    \frac{3}{2}A_1\left(\frac{\bar{I}_1}{3} - 1\right)^{P_1} + \
    \frac{3}{2}C_1\left(\frac{\bar{I}_1}{3} - 1\right)^{R_1}
    $$
    """

    bulk_modulus: Property
    A1: Property
    P1: Property
    B1: Property
    Q1: Property
    C1: Property
    R1: Property
    # hack because Swanson is a stupid model
    cutoff_strain: float = eqx.field(static=True)

    def energy(self, grad_u, theta, state_old, dt):
        K = self.bulk_modulus
        A1, P1 = self.A1, self.P1
        B1, Q1 = self.B1, self.Q1
        C1, R1 = self.C1, self.R1
        tau_cutoff = (1.0 / 3.0) * (3.0 + self.cutoff_strain**2) - 1.0

        # kinematics
        J = self.jacobian(grad_u)
        I_1_bar = self.I1_bar(grad_u)
        I_2_bar = self.I2_bar(grad_u)
        tau_1 = (1.0 / 3.0) * I_1_bar - 1.0
        tau_2 = (1.0 / 3.0) * I_2_bar - 1.0
        tau_tilde_1 = tau_1 + tau_cutoff
        tau_tilde_2 = tau_2 + tau_cutoff

        # constitutive
        W_vol = K * (J * jnp.log(J) - J + 1.0)
        W_dev_tau = (
            3.0
            / 2.0
            * (
                A1 / (P1 + 1.0) * (tau_tilde_1 ** (P1 + 1.0))
                + B1 / (Q1 + 1.0) * (tau_tilde_2 ** (Q1 + 1.0))
                + C1 / (R1 + 1.0) * (tau_tilde_1 ** (R1 + 1.0))
            )
        )
        W_dev_cutoff = (
            3.0
            / 2.0
            * (
                A1 / (P1 + 1.0) * (tau_cutoff ** (P1 + 1.0))
                + B1 / (Q1 + 1.0) * (tau_cutoff ** (Q1 + 1.0))
                + C1 / (R1 + 1.0) * (tau_cutoff ** (R1 + 1.0))
            )
        )
        W_dev = W_dev_tau - W_dev_cutoff
        return W_vol + W_dev, state_old
