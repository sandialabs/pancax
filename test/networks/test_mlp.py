def test_linear():
    from pancax import Linear
    import jax
    model = Linear(3, 2, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)


def test_mlp():
    from pancax import MLP
    import jax
    model = MLP(3, 2, 20, 3, jax.nn.tanh, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)


def test_mlp_basis():
    from pancax import MLPBasis
    import jax
    model = MLPBasis(3, 20, 3, jax.nn.tanh, key=jax.random.key(0))
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (20,)
