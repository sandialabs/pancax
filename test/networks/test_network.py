def test_network():
    from pancax import MLP, Network
    import jax
    model = Network(MLP, 3, 2, 20, 3, jax.nn.tanh, key=jax.random.key(0)) 
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)
