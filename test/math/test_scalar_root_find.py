# from sys import float_info
# import jax
# import jax.numpy as jnp
# import pancax.math.scalar_root_find as ScalarRootFind
import pytest

# from scipy.optimize import minimize_scalar, root_scalar # for comparison
# from scipy.optimize import OptimizeResult

# from optimism import ScalarRootFind
# from . import TestFixture


def f(x):
    return x**3 - 4.0


@pytest.fixture
def srf_fixture():
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings = ScalarRootFind.get_settings(r_tol=1.e-12, x_tol=0.)
    root_guess = 1.e-5
    root_expected = jnp.cbrt(4.0)
    return settings, root_guess, root_expected


def test_find_root(srf_fixture):
    from sys import float_info
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    rootBracket = jnp.array([float_info.epsilon, 100.0])
    root, _ = ScalarRootFind.find_root(f, root_guess, rootBracket, settings)
    converged = jnp.abs(f(root)) <= settings.r_tol
    assert converged
    assert jnp.allclose(root, root_expected, rtol=1.e-13)


def test_find_root_with_jit(srf_fixture):
    from sys import float_info
    import jax
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    rtsafe_jit = jax.jit(ScalarRootFind.find_root, static_argnums=(0, 3))
    rootBracket = jnp.array([float_info.epsilon, 100.0])
    root, _ = rtsafe_jit(f, root_guess, rootBracket, settings)
    converged = jnp.abs(f(root)) <= settings.r_tol
    assert converged
    assert jnp.allclose(root, root_expected, rtol=1.e-13)


def test_unbracketed_root_gives_nan(srf_fixture):
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    root_bracket = jnp.array([2.0, 100.0])
    root, _ = ScalarRootFind.find_root(f, root_guess, root_bracket, settings)
    assert jnp.isnan(root)


def test_find_root_converges_on_hard_function(srf_fixture):
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    # g = lambda x: jnp.sin(x) + x
    def g(x):
        return jnp.sin(x) + x

    root_bracket = jnp.array([-3.0, 20.0])
    x0 = 19.0
    root, _ = ScalarRootFind.find_root(f, x0, root_bracket, settings)
    converged = jnp.abs(f(root)) <= settings.r_tol
    assert converged
    assert jnp.allclose(root, root_expected, rtol=1.e-13)


def test_root_find_is_differentiable(srf_fixture):
    from sys import float_info
    import jax
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    # myfunc = lambda x, a: x**3 - a
    def myfunc(x, a):
        return x**3 - a

    def cube_root(a):
        rootBracket = jnp.array([float_info.epsilon, a])
        root, _ = ScalarRootFind.find_root(
            lambda x: myfunc(x, a), 8.0, rootBracket,
            settings
        )
        return root

    root = cube_root(4.0)
    assert jnp.allclose(root, root_expected, rtol=1.e-13)

    df = jax.jacfwd(cube_root)
    x = 3.0
    assert jnp.allclose(df(x), x**(-2 / 3) / 3, rtol=1.e-13)


def test_find_root_with_forced_bisection_step(srf_fixture):
    from sys import float_info
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    # myfunc = lambda x, a: x**2 - a
    def myfunc(x, a):
        return x**2 - a

    def my_sqrt(a):
        root_bracket = jnp.array([float_info.epsilon, a])
        root, _ = ScalarRootFind.find_root(
            lambda x: myfunc(x, a), 8.0,
            root_bracket,
            settings
        )
        return root

    r = my_sqrt(9.0)
    assert jnp.allclose(r, 3., rtol=1.e-12)


def test_root_find_with_vmap_and_jit(srf_fixture):
    from sys import float_info
    import jax
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    # myfunc = lambda x, a: x**2 - a
    def myfunc(x, a):
        return x**2 - a

    def my_sqrt(a):
        root_bracket = jnp.array([float_info.epsilon, a])
        root, _ = ScalarRootFind.find_root(
            lambda x: myfunc(x, a), 8.0,
            root_bracket, settings
        )
        return root

    x = jnp.array([1.0, 4.0, 9.0, 16.0])
    F = jax.jit(jax.vmap(my_sqrt, 0))
    expected_roots = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(F(x), expected_roots, rtol=1.e-12)


def test_solves_when_left_bracket_is_solution(srf_fixture):
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    root_bracket = jnp.array([0.0, 1.0])
    guess = 3.0
    # f = lambda x: x*(x**2 - 10.0)

    def f(x):
        return x * (x**2 - 10.)

    root, _ = ScalarRootFind.find_root(f, guess, root_bracket, settings)
    converged = jnp.abs(f(root)) <= settings.r_tol
    assert converged
    assert jnp.allclose(root, 0., rtol=1.e-12)


def test_solves_when_right_bracket_is_solution(srf_fixture):
    import jax.numpy as jnp
    import pancax.math.scalar_root_find as ScalarRootFind

    settings, root_guess, root_expected = srf_fixture

    root_bracket = jnp.array([-1.0, 0.0])
    guess = 3.0
    # f = lambda x: x*(x**2 - 10.0)

    def f(x):
        return x * (x**2 - 10.)

    root, _ = ScalarRootFind.find_root(f, guess, root_bracket, settings)
    converged = jnp.abs(f(root)) <= settings.r_tol
    # self.assertTrue(converged)
    # self.assertNear(root, 0.0, 12)
    assert converged
    assert jnp.allclose(root, 0., rtol=1.e-12)
