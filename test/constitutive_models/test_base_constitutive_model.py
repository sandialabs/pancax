from pancax import NeoHookean, NeoHookeanFixedBulkModulus
import jax
import jax.numpy as jnp


def test_jacobian():
    model = NeoHookean()
    F = jnp.array([
        [4., 0., 0.],
        [0., 2., 0.],
        [0., 0., 1.]
    ])
    J = model.jacobian(F)
    assert jnp.array_equal(J, jnp.linalg.det(F))

    # TODO add better test.
    # F = jax.random.uniform((3, 3), key=jax.random.key(0))
    # J = model.jacobian(F)
    # assert jnp.array_equal(J, jnp.linalg.det(F))


def test_jacobian_bad_value():
    model = NeoHookean()
    F = jnp.array([
        [4., 0., 0.],
        [0., 2., 0.],
        [0., 0., -1.]
    ])
    J = model.jacobian(F)
    assert jnp.array_equal(J, 1.e3)


def test_bulk_modulus_init():
    model = NeoHookeanFixedBulkModulus(bulk_modulus=10.0)
    assert model.bulk_modulus == 10.0
