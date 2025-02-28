from pancax import NeoHookean
import jax.numpy as jnp


K = 0.833
G = 0.3846


def test_jacobian():
    model = NeoHookean(bulk_modulus=K, shear_modulus=G)
    F = jnp.array([
        [4., 0., 0.],
        [0., 2., 0.],
        [0., 0., 1.]
    ])
    grad_u = F - jnp.eye(3)
    J = model.jacobian(grad_u)
    assert jnp.array_equal(J, jnp.linalg.det(F))

    # TODO add better test.
    # F = jax.random.uniform((3, 3), key=jax.random.key(0))
    # J = model.jacobian(F)
    # assert jnp.array_equal(J, jnp.linalg.det(F))


def test_jacobian_bad_value():
    model = NeoHookean(bulk_modulus=K, shear_modulus=G)
    F = jnp.array([
        [4., 0., 0.],
        [0., 2., 0.],
        [0., 0., -1.]
    ])
    grad_u = F - jnp.eye(3)
    J = model.jacobian(grad_u)
    assert jnp.array_equal(J, 1.e3)
