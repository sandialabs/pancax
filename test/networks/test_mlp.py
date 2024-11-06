from pancax import Linear, MLP, MLPBasis
import jax


def test_linear():
    model = Linear(3, 2, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)


def test_mlp():
    model = MLP(3, 2, 20, 3, jax.nn.tanh, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)


def test_mlp_basis():
    model = MLPBasis(3, 20, 3, jax.nn.tanh, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (20,)
