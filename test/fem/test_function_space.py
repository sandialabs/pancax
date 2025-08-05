import pytest


# coords = np.array([[-0.6162298 ,  4.4201174],
#                    [-2.2743905 ,  4.53892   ],
#                    [ 2.0868123 ,  0.68486094]])
# conn = np.arange(0, 3)
# parentElement = Interpolants.make_parent_element_2d(degree=1)


# mesh
Nx = 7
Ny = 7
xRange = [0.0, 1.0]
yRange = [0.0, 1.0]
# targetDispGrad = jnp.array([[0.1, -0.2],[0.4, -0.1]])

# mesh, U = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: jnp.dot(targetDispGrad, x))


@pytest.fixture
def mesh_and_disp():
    from .utils import create_mesh_and_disp
    import jax.numpy as jnp

    targetDispGrad = jnp.array([[0.1, -0.2], [0.4, -0.1]])
    mesh, U = create_mesh_and_disp(
        Nx, Ny, xRange, yRange, lambda x: jnp.dot(targetDispGrad, x)
    )
    return mesh, U, targetDispGrad


# function space
@pytest.fixture
def fspace_fixture_1(mesh_and_disp):
    from pancax.fem import NonAllocatedFunctionSpace, QuadratureRule
    from pancax.fem import construct_function_space
    import jax.numpy as jnp

    mesh, _, _ = mesh_and_disp
    quadratureRule = QuadratureRule(mesh.parentElement, 1)
    fs = construct_function_space(mesh, quadratureRule)
    fs_na = NonAllocatedFunctionSpace(mesh, quadratureRule)
    nElements = mesh.num_elements
    nQuadPoints = len(quadratureRule)
    state = jnp.zeros((nElements, nQuadPoints, 1))
    props = jnp.array([1.0, 1.0])
    dt = 0.0
    return fs, fs_na, quadratureRule, state, props, dt


@pytest.fixture
def fspace_fixture_2(mesh_and_disp):
    from pancax.fem import NonAllocatedFunctionSpace, QuadratureRule
    from pancax.fem import construct_function_space
    import jax.numpy as jnp

    mesh, _, _ = mesh_and_disp
    quadratureRule = QuadratureRule(mesh.parentElement, 1)
    fs = construct_function_space(mesh, quadratureRule)
    fs_na = NonAllocatedFunctionSpace(mesh, quadratureRule)
    nElements = mesh.num_elements
    nQuadPoints = len(quadratureRule)
    state = jnp.zeros((nElements, nQuadPoints, 1))
    props = jnp.array([1.0, 1.0])
    dt = 0.0
    return fs, fs_na, quadratureRule, state, props, dt


def test_element_volume_single_point_quadrature(fspace_fixture_1, mesh_and_disp):
    import jax.numpy as jnp

    fs, _, _, _, _, _ = fspace_fixture_1
    mesh, _, _ = mesh_and_disp
    elementVols = jnp.sum(fs.vols, axis=1)
    nElements = mesh.num_elements
    jnp.array_equal(elementVols, jnp.ones(nElements) * 0.5 / ((Nx - 1) * (Ny - 1)))


def test_element_volume_single_point_quadrature_na(fspace_fixture_1, mesh_and_disp):
    import jax
    import jax.numpy as jnp

    _, fs_na, _, _, _, _ = fspace_fixture_1
    mesh, _, _ = mesh_and_disp
    X_els = mesh.coords[mesh.conns, :]
    elementVols = jnp.sum(jax.vmap(fs_na.JxWs)(X_els), axis=1)
    nElements = mesh.num_elements
    jnp.array_equal(elementVols, jnp.ones(nElements) * 0.5 / ((Nx - 1) * (Ny - 1)))


def test_linear_reproducing_single_point_quadrature(fspace_fixture_1, mesh_and_disp):
    from pancax.fem.function_space import compute_field_gradient
    import jax.numpy as jnp

    fs, _, quadratureRule, _, _, _ = fspace_fixture_1
    mesh, U, targetDispGrad = mesh_and_disp
    dispGrads = compute_field_gradient(fs, U, mesh.coords)
    nElements = mesh.num_elements
    npts = quadratureRule.xigauss.shape[0]
    exact = jnp.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert jnp.allclose(dispGrads, exact)


def test_linear_reproducing_single_point_quadrature_na(fspace_fixture_1, mesh_and_disp):
    import jax
    import jax.numpy as jnp

    _, fs_na, quadratureRule, _, _, _ = fspace_fixture_1
    mesh, U, targetDispGrad = mesh_and_disp
    X_els = mesh.coords[mesh.conns, :]
    U_els = U[mesh.conns, :]
    dispGrads = jax.vmap(fs_na.compute_field_gradient, in_axes=(0, 0))(U_els, X_els)
    nElements = mesh.num_elements
    npts = quadratureRule.xigauss.shape[0]
    exact = jnp.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert jnp.allclose(dispGrads, exact)


def test_integrate_constant_field_single_point_quadrature(
    fspace_fixture_1, mesh_and_disp
):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_1
    mesh, U, _ = mesh_and_disp

    integralOfOne = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
        mesh.blocks["block"],
    )
    jnp.isclose(integralOfOne, 1.0)


def test_integrate_constant_field_single_point_quadrature_na(
    fspace_fixture_1, mesh_and_disp
):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_1
    mesh, U, _ = mesh_and_disp
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]
    integralOfOne = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    jnp.isclose(integralOfOne, 1.0)


def test_integrate_linear_field_single_point_quadrature(
    fspace_fixture_1, mesh_and_disp
):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_1
    mesh, U, _ = mesh_and_disp
    Ix = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
        mesh.blocks["block"],
    )
    # displacement at x=1 should match integral
    idx = jnp.argmax(mesh.coords[:, 0])
    expected = U[idx, 0] * (yRange[1] - yRange[0])
    jnp.isclose(Ix, expected)

    Iy = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
        mesh.blocks["block"],
    )
    idx = jnp.argmax(mesh.coords[:, 1])
    expected = U[idx, 1] * (xRange[1] - xRange[0])
    jnp.isclose(Iy, expected)


def test_integrate_linear_field_single_point_quadrature_na(
    fspace_fixture_1, mesh_and_disp
):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_1
    mesh, U, _ = mesh_and_disp
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]

    Ix = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
    )
    idx = jnp.argmax(mesh.coords[:, 0])
    expected = U[idx, 0] * (yRange[1] - yRange[0])
    jnp.isclose(Ix, expected)

    Iy = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
    )

    idx = jnp.argmax(mesh.coords[:, 1])
    expected = U[idx, 1] * (xRange[1] - xRange[0])
    jnp.isclose(Iy, expected)


def test_element_volume_multi_point_quadrature(fspace_fixture_2, mesh_and_disp):
    import jax.numpy as jnp

    fs, _, _, _, _, _ = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    elementVols = jnp.sum(fs.vols, axis=1)
    nElements = mesh.num_elements
    jnp.array_equal(elementVols, jnp.ones(nElements) * 0.5 / ((Nx - 1) * (Ny - 1)))


def test_element_volume_multi_point_quadrature_na(fspace_fixture_2, mesh_and_disp):
    import jax
    import jax.numpy as jnp

    _, fs_na, _, _, _, _ = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    X_els = mesh.coords[mesh.conns, :]
    elementVols = jnp.sum(jax.vmap(fs_na.JxWs)(X_els), axis=1)
    nElements = mesh.num_elements
    jnp.array_equal(elementVols, jnp.ones(nElements) * 0.5 / ((Nx - 1) * (Ny - 1)))


def test_linear_reproducing_multi_point_quadrature(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import compute_field_gradient
    import jax.numpy as jnp

    fs, _, quadratureRule, _, _, _ = fspace_fixture_2
    mesh, U, targetDispGrad = mesh_and_disp
    dispGrads = compute_field_gradient(fs, U, mesh.coords)
    nElements = mesh.num_elements
    npts = quadratureRule.xigauss.shape[0]
    exact = jnp.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert jnp.allclose(dispGrads, exact)


def test_linear_reproducing_multi_point_quadrature_na(fspace_fixture_2, mesh_and_disp):
    import jax
    import jax.numpy as jnp

    _, fs_na, quadratureRule, _, _, _ = fspace_fixture_2
    mesh, U, targetDispGrad = mesh_and_disp
    X_els = mesh.coords[mesh.conns, :]
    U_els = U[mesh.conns, :]
    dispGrads = jax.vmap(fs_na.compute_field_gradient, in_axes=(0, 0))(U_els, X_els)
    nElements = mesh.num_elements
    npts = quadratureRule.xigauss.shape[0]
    exact = jnp.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert jnp.allclose(dispGrads, exact)


def test_integrate_constant_field_multi_point_point_quadrature(
    fspace_fixture_2, mesh_and_disp
):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    integralOfOne = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
        mesh.blocks["block"],
    )
    jnp.isclose(integralOfOne, 1.0)


def test_integrate_constant_field_multi_point_quadrature_na(
    fspace_fixture_2, mesh_and_disp
):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]
    integralOfOne = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    jnp.isclose(integralOfOne, 1.0)


# TODO add integration test/method for new na fspace


def test_integrate_linear_field_multi_point_quadrature(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    Ix = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
        mesh.blocks["block"],
    )
    idx = jnp.argmax(mesh.coords[:, 0])
    expected = U[idx, 0] * (yRange[1] - yRange[0])
    jnp.isclose(Ix, expected)

    Iy = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
        mesh.blocks["block"],
    )
    idx = jnp.argmax(mesh.coords[:, 1])
    expected = U[idx, 1] * (xRange[1] - xRange[0])
    jnp.isclose(Iy, expected)


def test_integrate_linear_field_multi_point_quadrature_na(
    fspace_fixture_2, mesh_and_disp
):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]

    Ix = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
    )
    idx = jnp.argmax(mesh.coords[:, 0])
    expected = U[idx, 0] * (yRange[1] - yRange[0])
    jnp.isclose(Ix, expected)

    Iy = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
    )

    idx = jnp.argmax(mesh.coords[:, 1])
    expected = U[idx, 1] * (xRange[1] - xRange[0])
    jnp.isclose(Iy, expected)


def test_integrate_over_half_block(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    mesh, U, _ = mesh_and_disp
    fs, _, _, state, props, dt = fspace_fixture_2
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0

    blockWithHalfTheVolume = slice(0, nElements // 2)
    integral = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
        blockWithHalfTheVolume,
    )
    jnp.isclose(integral, 1.0 / 2.0)


def test_integrate_over_half_block_na(fspace_fixture_2, mesh_and_disp):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0
    blockWithHalfTheVolume = slice(0, nElements // 2)

    U_els = U[mesh.conns[blockWithHalfTheVolume], :]
    X_els = U[mesh.conns[blockWithHalfTheVolume], :]

    integral = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state[blockWithHalfTheVolume, :, :],
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    jnp.isclose(integral, 1.0 / 2.0)


def test_integrate_over_half_block_indices(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import integrate_over_block
    import jax.numpy as jnp

    mesh, U, _ = mesh_and_disp
    fs, _, _, state, props, dt = fspace_fixture_2
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0

    blockWithHalfTheVolume = jnp.arange(nElements // 2)

    integral = integrate_over_block(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
        blockWithHalfTheVolume,
    )
    jnp.isclose(integral, 1.0 / 2.0)


def test_integrate_over_half_block_indices_na(fspace_fixture_2, mesh_and_disp):
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0

    blockWithHalfTheVolume = jnp.arange(nElements // 2)

    U_els = U[mesh.conns[blockWithHalfTheVolume], :]
    X_els = U[mesh.conns[blockWithHalfTheVolume], :]

    integral = fs_na.integrate_on_elements(
        U_els,
        X_els,
        state[blockWithHalfTheVolume, :, :],
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    jnp.isclose(integral, 1.0 / 2.0)


def test_jit_on_integration(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import integrate_over_block
    import jax
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    integrate_jit = jax.jit(integrate_over_block, static_argnums=(6,))
    I = integrate_jit(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
        mesh.blocks["block"],
    )
    jnp.isclose(I, 1.0)


def test_jit_on_integration_na(fspace_fixture_2, mesh_and_disp):
    import equinox as eqx
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    integrate_jit = eqx.filter_jit(fs_na.integrate_on_elements)
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]
    I = integrate_jit(
        U_els, X_els, state, props, dt, lambda u, gradu, state, props, X, dt: 1.0
    )
    jnp.isclose(I, 1.0)


def test_jit_and_jacrev_on_integration(fspace_fixture_2, mesh_and_disp):
    from pancax.fem.function_space import integrate_over_block
    import jax
    import jax.numpy as jnp

    fs, _, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    F = jax.jit(jax.jacrev(integrate_over_block, 1), static_argnums=(6,))
    dI = F(
        fs,
        U,
        mesh.coords,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 0.5 * jnp.tensordot(gradu, gradu),
        mesh.blocks["block"],
    )
    nNodes = mesh.coords.shape[0]
    interiorNodeIds = jnp.setdiff1d(jnp.arange(nNodes), mesh.nodeSets["all_boundary"])
    jnp.array_equal(dI[interiorNodeIds, :], jnp.zeros_like(U[interiorNodeIds, :]))


def test_jit_and_jacrev_on_integration_na(fspace_fixture_2, mesh_and_disp):
    import equinox as eqx
    import jax.numpy as jnp

    _, fs_na, _, state, props, dt = fspace_fixture_2
    mesh, U, _ = mesh_and_disp
    F = eqx.filter_jit(eqx.filter_jacrev(fs_na.integrate_on_elements))
    U_els = U[mesh.conns[mesh.blocks["block"]], :]
    X_els = U[mesh.conns[mesh.blocks["block"]], :]
    dI = F(
        U_els,
        X_els,
        state,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 0.5 * jnp.tensordot(gradu, gradu),
    )
    nNodes = mesh.coords.shape[0]
    interiorNodeIds = jnp.setdiff1d(jnp.arange(nNodes), mesh.nodeSets["all_boundary"])
    jnp.array_equal(dI[interiorNodeIds, :], jnp.zeros_like(U[interiorNodeIds, :]))
