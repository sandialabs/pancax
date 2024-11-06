from .base_constitutive_model import *
import jax.numpy as jnp


class Swanson4(ConstitutiveModel):
  r"""
	Swanson model truncated to 4 parameters

	.. math::
		\psi(\mathbf{F}) = K\left(J\ln J - J + 1\right) +
							         \frac{3}{2}A_1\left(\frac{\bar{I}_1}{3} - 1\right)^{P_1} + 
                       \frac{3}{2}C_1\left(\frac{\bar{I}_1}{3} - 1\right)^{R_1}
	"""
  n_properties = 5
  property_names = [
    'bulk modulus', 'A1', 'P1', 'C1', 'R1'
  ]
  cutoff_strain: float

  def __init__(self, cutoff_strain):
    self.cutoff_strain = cutoff_strain

  def energy(self, F, props):
		# unpack props
    K, A1, P1, C1, R1 = self.unpack_properties(props)
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

  def unpack_properties(self, props):
    return props[0], props[1], props[2], props[3], props[4]


class Swanson4FixedBulkModulus(Swanson4, ConstitutiveModelFixedBulkModulus):
  n_properties = 4
  property_names = [
    'A1', 'P1', 'C1', 'R1'
  ]
  bulk_modulus: float

  def __init__(self, bulk_modulus: float, cutoff_strain: float):
    super().__init__(cutoff_strain)
    self.bulk_modulus = bulk_modulus

  def unpack_properties(self, props):
    return self.bulk_modulus, props[0], props[1], props[2], props[3]


class Swanson6(ConstitutiveModel):
  r"""
  Swanson model truncated to 6 parameters

	.. math::
		\psi(\mathbf{F}) = K\left(J\ln J - J + 1\right) +
							         \frac{3}{2}A_1\left(\frac{\bar{I}_1}{3} - 1\right)^{P_1} + 
                       \frac{3}{2}B_1\left(\frac{\bar{I}_2}{3} - 1\right)^{Q_1} + 
                       \frac{3}{2}C_1\left(\frac{\bar{I}_1}{3} - 1\right)^{R_1}
  """
  n_properties = 7
  property_names = [
    'bulk modulus', 'A1', 'P1', 'B1', 'Q1', 'C1', 'R1'
  ]
  cutoff_strain: float

  def __init__(self, cutoff_strain: float):
    self.cutoff_strain = cutoff_strain

  def energy(self, F, props):
		# unpack props
    K, A1, P1, B1, Q1, C1, R1 = self.unpack_properties(props)
    tau_cutoff = (1. / 3.) * (3. + self.cutoff_strain**2) - 1.

		# kinematics
    J = self.jacobian(F)
    C = F.T @ F
    C_bar = jnp.power(J, -2. / 3.) * C
    C_bar_2 = C_bar @ C_bar

    I_1_bar = jnp.trace(C_bar)
    I_2_bar = 0.5 * (I_1_bar**2 - jnp.trace(C_bar_2))
    tau_1 = (1. / 3.) * I_1_bar - 1.
    tau_2 = (1. / 3.) * I_2_bar - 1.
    tau_tilde_1 = tau_1 + tau_cutoff
    tau_tilde_2 = tau_2 + tau_cutoff

    # constitutive
    W_vol = K * (J * jnp.log(J) - J + 1.)
    W_dev_tau = 3. / 2. * (
      A1 / (P1 + 1.) * (tau_tilde_1**(P1 + 1.)) +
      B1 / (Q1 + 1.) * (tau_tilde_2**(Q1 + 1.)) +
      C1 / (R1 + 1.) * (tau_tilde_1**(R1 + 1.))
    )
    W_dev_cutoff = 3. / 2. * (
      A1 / (P1 + 1.) * (tau_cutoff**(P1 + 1.)) +
      B1 / (Q1 + 1.) * (tau_cutoff**(Q1 + 1.)) +
      C1 / (R1 + 1.) * (tau_cutoff**(R1 + 1.))
    )
    W_dev = W_dev_tau - W_dev_cutoff
    return W_vol + W_dev
  
  def unpack_properties(self, props):
    return props[0], props[1], props[2], props[3], props[4], props[5], props[6]
  

class Swanson6FixedBulkModulus(Swanson6, ConstitutiveModelFixedBulkModulus):
  n_properties = 6
  property_names = [
    'A1', 'P1', 'B1', 'Q1', 'C1', 'R1'
  ]
  bulk_modulus: float
  cutoff_strain: float

  def __init__(self, bulk_modulus: float, cutoff_strain: float):
    super().__init__(cutoff_strain)
    self.bulk_modulus = bulk_modulus


  def unpack_properties(self, props):
    return self.bulk_modulus, props[0], props[1], props[2], props[3], props[4], props[5]
