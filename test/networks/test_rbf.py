def test_rbf_basis():
    from pancax.networks.rbf import RBFBasis
    import jax
    key = jax.random.PRNGKey(0)
    model = RBFBasis(4, 40, key=key)
    x = jax.random.uniform(key=key, shape=(4,))
    y = model(x)
    assert y.shape == (40,)
