from .base import BaseConstitutiveModel
from .properties import Property
import equinox as eqx
import jax.numpy as jnp


class Swanson(BaseConstitutiveModel):
  r"""
	Swanson model truncated to 4 parameters

	$$
  \psi(\mathbf{F}) = K\left(J\ln J - J + 1\right) +
                     \frac{3}{2}A_1\left(\frac{\bar{I}_1}{3} - 1\right)^{P_1} + 
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

  def energy(self, F):
    K = self.bulk_modulus
    A1, P1 = self.A1, self.P1
    B1, Q1 = self.B1, self.Q1
    C1, R1 = self.C1, self.R1
    tau_cutoff = (1. / 3.) * (3. + self.cutoff_strain**2) - 1.

    # kinematics
    J = self.jacobian(F)
    C = F.T @ F
    C_bar = jnp.power(J, -2. / 3.) * C
    I_1_bar = jnp.trace(C_bar)
    tau_1 = (1. / 3.) * I_1_bar - 1.
    tau_tilde_1 = tau_1 + tau_cutoff

    # constitutive
    W_vol = K * (J * jnp.log(J) - J + 1.)
    W_dev_tau = 3. / 2. * (
      A1 / (P1 + 1.) * (tau_tilde_1**(P1 + 1.)) +
      C1 / (R1 + 1.) * (tau_tilde_1**(R1 + 1.))
    )
    W_dev_cutoff = 3. / 2. * (
      A1 / (P1 + 1.) * (tau_cutoff**(P1 + 1.)) +
      C1 / (R1 + 1.) * (tau_cutoff**(R1 + 1.))
    )
    W_dev = W_dev_tau - W_dev_cutoff
    return W_vol + W_dev
