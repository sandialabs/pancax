from .base_constitutive_model import ConstitutiveModel
import jax.numpy as jnp


class BlatzKo(ConstitutiveModel):
    r"""
	Blatz-Ko model with the following model form

	.. math::
		\psi(\mathbf{F}) = \frac{1}{2}\mu\left(\frac{I_2}{I_3} + 2\sqrt{I_3} - 5\right)
	"""
    n_properties = 1
    property_names = [
        'shear modulus'
    ]

    def energy(self, F, props):
        G = self.unpack_properties(props)

        # kinematics
        I2 = self.I2(F)
        I3 = jnp.linalg.det(F.T @ F)

        # constitutive
        W = (G / 2.) * (I2 / I3 + 2 * jnp.sqrt(I3) - 5.)
        return W

    def unpack_properties(self, props):
        return props[0]
