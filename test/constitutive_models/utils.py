def uniaxial_strain(lambda_: float):
    import jax.numpy as jnp
    return jnp.array([[lambda_, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def simple_shear(gamma: float):
    import jax.numpy as jnp
    return jnp.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
