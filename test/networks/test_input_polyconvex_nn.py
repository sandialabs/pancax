def is_convex(network, x1, x2, lambda_val):
    """Check if the network is convex between two points x1 and x2."""
    # Calculate the convex combination
    x_combined = lambda_val * x1 + (1 - lambda_val) * x2

    # Evaluate the network at the points
    f_x1 = network(x1)
    f_x2 = network(x2)
    f_combined = network(x_combined)

    # Check the convexity condition
    return f_combined <= lambda_val * f_x1 + (1 - lambda_val) * f_x2


def is_polyconvex(network, fixed_inputs, dim, lambda_val):
    import jax.random as random
    """Check polyconvexity for a specific dimension."""
    # Create two input points with fixed values in other dimensions
    x1 = fixed_inputs.copy()
    x2 = fixed_inputs.copy()

    # Vary the specified dimension
    x1 = x1.at[dim].set(random.uniform(random.PRNGKey(0), ()))
    x2 = x2.at[dim].set(random.uniform(random.PRNGKey(1), ()))

    return is_convex(network, x1, x2, lambda_val)


def test_icnn_all_convex():
    from pancax import InputPolyconvexNN
    import jax
    model = InputPolyconvexNN(
        3, 1, 3, jax.nn.softplus, jax.nn.softplus,
        key=jax.random.key(0)
    )
    model = model.parameter_enforcement()

    x1 = jax.random.uniform(key=jax.random.key(0), shape=(3,))
    x2 = jax.random.uniform(key=jax.random.key(1), shape=(3,))
    # y = model(x)

    for dim in range(x1.shape[0]):
        for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert is_polyconvex(model, x1, dim, lambda_val), \
                f"Polyconvexity failed for dimension={dim}, \
                    lambda={lambda_val}, fixed_inputs={x1}"

    for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert is_convex(model, x1, x2, lambda_val), f"Convexity failed for \
            lambda={lambda_val}, x1={x1}, x2={x2}"


def test_icnn_some_convex():
    from pancax import InputPolyconvexNN
    import jax
    n_convex = 2
    model = InputPolyconvexNN(
        3, 1, n_convex, jax.nn.softplus, jax.nn.softplus,
        key=jax.random.key(0)
    )
    model = model.parameter_enforcement()

    x1 = jax.random.uniform(key=jax.random.key(0), shape=(3,))
    # y = model(x)

    for dim in range(n_convex):
        for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert is_polyconvex(model, x1, dim, lambda_val), \
                f"Polyconvexity failed for dimension={dim}, \
                    lambda={lambda_val}, fixed_inputs={x1}"
