from .base_constitutive_model import ConstitutiveModel, ConstitutiveModelFixedBulkModulus
import jax
import jax.numpy as jnp


class Gent(ConstitutiveModel):
    r"""
	Gent model with the following model form

	.. math::
		\psi(\mathbf{F}) = \frac{1}{2}K\left[\frac{1}{2}\left(J^2 - \ln J\right)\right] -
						   \frac{1}{2}GJ_m\ln\left(1 - \frac{\bar{I}_1 - 3}{J_m}\right)
	"""
    n_properties = 3
    property_names = [
        'bulk modulus',
        'shear modulus',
        'Jm parameter'
    ]

    def energy(self, F, props):
        # unpack properties
        K, G, Jm = self.unpack_properties(props)

        # kinematics
        C = F.T @ F
        J = self.jacobian(F)
        I_1_bar = jnp.trace(jnp.power(J, -2. / 3.) * C)

        # guard rail
        check_value = I_1_bar > Jm + 3.0 - 0.001
        I_1_bar = jax.lax.cond(
            check_value,
            lambda x: Jm + 3.0 - 0.001,
            lambda x: x,
            I_1_bar
        )

        # constitutive
        W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
        W_dev = -0.5 * G * Jm * jnp.log(1. - ((I_1_bar - 3.0) / Jm))
        return W_vol + W_dev
    
    def unpack_properties(self, props):
        return props[0], props[1], props[2]
    

class GentFixedBulkModulus(Gent, ConstitutiveModelFixedBulkModulus):
    n_properties = 2
    property_names = [
        'shear modulus',
        'Jm parameter'
    ]
    bulk_modulus: float

    def unpack_properties(self, props):
        return self.bulk_modulus, props[0], props[1]
