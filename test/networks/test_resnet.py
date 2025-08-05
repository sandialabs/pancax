def test_resnet():
    from pancax import ResNet
    import jax
    key = jax.random.PRNGKey(0)
    model = ResNet(3, 2, 30, 3, jax.nn.tanh, key=key)
    x = jax.numpy.ones(3)
    y = model(x)
    assert y.shape == (2,)
