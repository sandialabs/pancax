import jax.numpy as jnp


def uniaxial_strain(lambda_: float):
    return jnp.array([[lambda_, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def simple_shear(gamma: float):
    return jnp.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
