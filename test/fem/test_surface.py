import pytest


@pytest.fixture
def surf_mesh_fix():
    from pancax.fem import LineElement, QuadratureRule
    from .utils import create_mesh_and_disp
    import jax.numpy as jnp
    Nx = 4
    Ny = 4
    L = 1.2
    W = 1.5
    xRange = [0.0, L]
    yRange = [0.0, W]

    targetDispGrad = jnp.zeros((2, 2))

    mesh, U = create_mesh_and_disp(
        Nx, Ny, xRange, yRange, lambda x: targetDispGrad.dot(x)
    )
    quad_rule = QuadratureRule(LineElement(1), 2)
    return mesh, U, L, W, quad_rule


def test_integrate_perimeter(surf_mesh_fix):
    from pancax.fem.surface import integrate_function_on_surface
    import jax.numpy as jnp
    mesh, _, L, W, quad_rule = surf_mesh_fix
    p = integrate_function_on_surface(
        quad_rule, mesh.sideSets["all_boundary"], mesh, lambda x, n: 1.0
    )
    # assertNear(p, 2*(L+W), 14)
    assert jnp.abs(p - 2 * (L + W)) < 1.0e-14


def test_integrate_quadratic_fn_on_surface(surf_mesh_fix):
    from pancax.fem.surface import integrate_function_on_surface
    import jax.numpy as jnp
    mesh, _, L, _, quad_rule = surf_mesh_fix
    Ival = integrate_function_on_surface(
        quad_rule, mesh.sideSets["top"], mesh, lambda x, n: x[0] ** 2
    )
    # assertNear(I, L**3/3.0, 14)
    assert jnp.abs(Ival - L ** (3 / 3.0)) < 1.0e14


def test_integrate_function_on_surface_that_uses_coords_and_normal(
    surf_mesh_fix
):
    from pancax.fem.surface import integrate_function_on_surface
    import jax.numpy as jnp
    mesh, _, L, W, quad_rule = surf_mesh_fix
    Ival = integrate_function_on_surface(
        quad_rule, mesh.sideSets["all_boundary"], mesh,
        lambda x, n: jnp.dot(x, n)
    )
    assert jnp.abs(Ival - 2 * L * W) < 1e-14
