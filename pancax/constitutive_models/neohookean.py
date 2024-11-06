from .base_constitutive_model import ConstitutiveModel, ConstitutiveModelFixedBulkModulus
import jax.numpy as jnp


class NeoHookean(ConstitutiveModel):
	r"""
	NeoHookean model with the following model form

	.. math::
		\psi(\mathbf{F}) = \frac{1}{2}K\left[\frac{1}{2}\left(J^2 - \ln J\right)\right] +
						   \frac{1}{2}G\left(\bar{I}_1 - 3\right)
	"""
	n_properties = 2
	property_names = [
		'bulk modulus',
		'shear modulus'
	]

	def energy(self, F, props):
		# unpack props
		K, G = self.unpack_properties(props)

		# kinematics
		C = F.T @ F
		J = self.jacobian(F)
		# I_1_bar = jnp.trace(jnp.power(J, -2. / 3.) * C)
		I_1_bar = jnp.trace(1. / jnp.square(jnp.cbrt(J)) * C)

		# constitutive
		W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
		W_dev = 0.5 * G * (I_1_bar - 3.)

		return W_vol + W_dev

	def unpack_properties(self, props):
		return props[0], props[1]


class NeoHookeanFixedBulkModulus(NeoHookean, ConstitutiveModelFixedBulkModulus):
	n_properties = 1
	property_names = [
		'shear modulus'
	]
	bulk_modulus: float

	def unpack_properties(self, props):
		return self.bulk_modulus, props[0]
