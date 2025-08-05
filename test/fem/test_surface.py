from pancax.fem import LineElement, QuadratureRule
from pancax.fem.surface import integrate_function_on_surface
from .utils import create_mesh_and_disp
import jax.numpy as jnp


Nx = 4
Ny = 4
L = 1.2
W = 1.5
xRange = [0.0, L]
yRange = [0.0, W]

targetDispGrad = jnp.zeros((2, 2))

mesh, U = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: targetDispGrad.dot(x))


quadRule = QuadratureRule(LineElement(1), 2)


def test_integrate_perimeter():
    print(mesh)
    p = integrate_function_on_surface(
        quadRule, mesh.sideSets["all_boundary"], mesh, lambda x, n: 1.0
    )
    # assertNear(p, 2*(L+W), 14)
    assert jnp.abs(p - 2 * (L + W)) < 1.0e-14


def test_integrate_quadratic_fn_on_surface():
    I = integrate_function_on_surface(
        quadRule, mesh.sideSets["top"], mesh, lambda x, n: x[0] ** 2
    )
    # assertNear(I, L**3/3.0, 14)
    assert jnp.abs(I - L ** (3 / 3.0)) < 1.0e14


def test_integrate_function_on_surface_that_uses_coords_and_normal():
    I = integrate_function_on_surface(
        quadRule, mesh.sideSets["all_boundary"], mesh, lambda x, n: jnp.dot(x, n)
    )
    assert jnp.abs(I - 2 * L * W) < 1e-14
