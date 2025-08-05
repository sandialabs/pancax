from pancax.math import math
import jax
import jax.numpy as jnp


def test_safe_sqrt():
    f = math.safe_sqrt(4.0)
    assert jnp.allclose(f, 2.0)

    df = jax.grad(math.safe_sqrt)(4.0)
    assert jnp.allclose(df, 0.25)

    f = math.safe_sqrt(-4.0)
    assert jnp.isnan(f)

    df = jax.grad(math.safe_sqrt)(-4.0)
    assert jnp.allclose(df, 0.0)


def test_sum2():
    a = jnp.arange(101)
    assert jnp.allclose(math.sum2(a), 100 * (100 + 1) / 2)


def test_dot2():
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    a = jax.random.uniform(key1, (100,), minval=1e-8, maxval=10.0)
    b = jax.random.uniform(key2, (100,), minval=1e-8, maxval=10.0)

    assert jnp.allclose(math.dot2(a, b), jnp.dot(a, b))
