from jax import vmap
from pancax.fem import QuadratureRule
from pancax.fem.function_space import compute_field_gradient
from pancax.fem.function_space import construct_function_space
from pancax.fem.function_space import integrate_over_block
from pancax.fem.function_space import NonAllocatedFunctionSpace
from pancax.fem.mesh import construct_mesh_from_basic_data
from pancax.fem.mesh import create_structured_mesh_data
from pancax.fem.surface import create_edges
import equinox as eqx
import jax
import jax.numpy as np


def create_mesh_and_disp(Nx, Ny, xRange, yRange, initial_disp_func, setNamePostFix=''):
    coords, conns = create_structured_mesh_data(Nx, Ny, xRange, yRange)
    tol = 1e-8
    nodeSets = {}
    nodeSets['left'+setNamePostFix] = np.flatnonzero(coords[:,0] < xRange[0] + tol)
    nodeSets['bottom'+setNamePostFix] = np.flatnonzero(coords[:,1] < yRange[0] + tol)
    nodeSets['right'+setNamePostFix] = np.flatnonzero(coords[:,0] > xRange[1] - tol)
    nodeSets['top'+setNamePostFix] = np.flatnonzero(coords[:,1] > yRange[1] - tol)
    nodeSets['all_boundary'+setNamePostFix] = np.flatnonzero(
        (coords[:,0] < xRange[0] + tol) |
        (coords[:,1] < yRange[0] + tol) |
        (coords[:,0] > xRange[1] - tol) |
        (coords[:,1] > yRange[1] - tol) 
    )
    
    def is_edge_on_left(xyOnEdge):
        return np.all( xyOnEdge[:,0] < xRange[0] + tol  )

    def is_edge_on_bottom(xyOnEdge):
        return np.all( xyOnEdge[:,1] < yRange[0] + tol  )

    def is_edge_on_right(xyOnEdge):
        return np.all( xyOnEdge[:,0] > xRange[1] - tol  )
    
    def is_edge_on_top(xyOnEdge):
        return np.all( xyOnEdge[:,1] > yRange[1] - tol  )

    sideSets = {}
    sideSets['left'+setNamePostFix] = create_edges(coords, conns, is_edge_on_left)
    sideSets['bottom'+setNamePostFix] = create_edges(coords, conns, is_edge_on_bottom)
    sideSets['right'+setNamePostFix] = create_edges(coords, conns, is_edge_on_right)
    sideSets['top'+setNamePostFix] = create_edges(coords, conns, is_edge_on_top)
    
    allBoundaryEdges = np.vstack([s for s in sideSets.values()])
    sideSets['all_boundary'+setNamePostFix] = allBoundaryEdges

    blocks = {'block'+setNamePostFix: np.arange(conns.shape[0])}
    mesh = construct_mesh_from_basic_data(coords, conns, blocks, nodeSets, sideSets)
    return mesh, vmap(initial_disp_func)(mesh.coords)



# coords = np.array([[-0.6162298 ,  4.4201174],
#                    [-2.2743905 ,  4.53892   ],
#                    [ 2.0868123 ,  0.68486094]])
# conn = np.arange(0, 3)
# parentElement = Interpolants.make_parent_element_2d(degree=1)


# mesh
Nx = 7
Ny = 7
xRange = [0.,1.]
yRange = [0.,1.]
targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])        

mesh, U = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: np.dot(targetDispGrad, x))

# function space
quadratureRule_1 = QuadratureRule(mesh.parentElement, 1)
fs_1 = construct_function_space(mesh, quadratureRule_1)
fs_1_na = NonAllocatedFunctionSpace(mesh, quadratureRule_1)
nElements = mesh.num_elements
nQuadPoints_1 = len(quadratureRule_1)
state_1 = np.zeros((nElements, nQuadPoints_1, 1))
props = np.array([1.0, 1.0])
dt = 0.0


def test_element_volume_single_point_quadrature():
    elementVols = np.sum(fs_1.vols, axis=1)
    nElements = mesh.num_elements
    np.array_equal(elementVols, np.ones(nElements)*0.5/((Nx-1)*(Ny-1)))


def test_element_volume_single_point_quadrature_na():
    X_els = mesh.coords[mesh.conns, :]
    elementVols = np.sum(vmap(fs_1_na.JxWs)(X_els), axis=1)
    nElements = mesh.num_elements
    np.array_equal(elementVols, np.ones(nElements)*0.5/((Nx-1)*(Ny-1)))


def test_linear_reproducing_single_point_quadrature():
    dispGrads = compute_field_gradient(fs_1, U, mesh.coords)
    nElements = mesh.num_elements
    npts = quadratureRule_1.xigauss.shape[0]
    exact = np.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert np.allclose(dispGrads, exact)


def test_linear_reproducing_single_point_quadrature_na():
    X_els = mesh.coords[mesh.conns, :]
    U_els = U[mesh.conns, :]
    dispGrads = vmap(fs_1_na.compute_field_gradient, in_axes=(0, 0))(U_els, X_els)
    nElements = mesh.num_elements
    npts = quadratureRule_1.xigauss.shape[0]
    exact = np.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert np.allclose(dispGrads, exact)


def test_integrate_constant_field_single_point_quadrature():
    integralOfOne = integrate_over_block(fs_1,
                                         U,
                                         mesh.coords,
                                         state_1,
                                         props,
                                         dt,
                                         lambda u, gradu, state, props, X, dt: 1.0,
                                         mesh.blocks['block'])
    np.isclose(integralOfOne, 1.0)


def test_integrate_constant_field_single_point_quadrature_na():
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]
    integralOfOne = fs_1_na.integrate_on_elements(
        U_els,
        X_els,
        state_1,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    np.isclose(integralOfOne, 1.0)


def test_integrate_linear_field_single_point_quadrature():
    
    Ix = integrate_over_block(fs_1,
                              U,
                              mesh.coords,
                              state_1,
                              props,
                              dt,
                              lambda u, gradu, state, props, X, dt: gradu[0,0],
                              mesh.blocks['block'])
    # displacement at x=1 should match integral
    idx = np.argmax(mesh.coords[:,0])
    expected = U[idx,0]*(yRange[1] - yRange[0])
    np.isclose(Ix, expected)
    
    Iy = integrate_over_block(fs_1,
                              U,
                              mesh.coords,
                              state_1,
                              props,
                              dt,
                              lambda u, gradu, state, props, X, dt: gradu[1, 1],
                              mesh.blocks['block'])
    idx = np.argmax(mesh.coords[:,1])
    expected = U[idx,1]*(xRange[1] - xRange[0])
    np.isclose(Iy, expected)


def test_integrate_linear_field_single_point_quadrature_na():
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]

    Ix = fs_1_na.integrate_on_elements(
        U_els,
        X_els,
        state_1,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
    )
    idx = np.argmax(mesh.coords[:,0])
    expected = U[idx,0]*(yRange[1] - yRange[0])
    np.isclose(Ix, expected)

    Iy = fs_1_na.integrate_on_elements(
        U_els,
        X_els,
        state_1,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
    )

    idx = np.argmax(mesh.coords[:,1])
    expected = U[idx,1]*(xRange[1] - xRange[0])
    np.isclose(Iy, expected)


quadratureRule_2 = QuadratureRule(mesh.parentElement, 2)
fs_2 = construct_function_space(mesh, quadratureRule_2)
fs_2_na = NonAllocatedFunctionSpace(mesh, quadratureRule_2)
nQuadPoints_2 = len(quadratureRule_2)
state_2 = np.zeros((nElements, nQuadPoints_2, 1))


def test_element_volume_multi_point_quadrature():
    elementVols = np.sum(fs_2.vols, axis=1)
    nElements = mesh.num_elements
    np.array_equal(elementVols, np.ones(nElements)*0.5/((Nx-1)*(Ny-1)))


def test_element_volume_multi_point_quadrature_na():
    X_els = mesh.coords[mesh.conns, :]
    elementVols = np.sum(vmap(fs_2_na.JxWs)(X_els), axis=1)
    nElements = mesh.num_elements
    np.array_equal(elementVols, np.ones(nElements)*0.5/((Nx-1)*(Ny-1)))


def test_linear_reproducing_multi_point_quadrature():
    dispGrads = compute_field_gradient(fs_2, U, mesh.coords)
    nElements = mesh.num_elements
    npts = quadratureRule_2.xigauss.shape[0]
    exact = np.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert np.allclose(dispGrads, exact)


def test_linear_reproducing_multi_point_quadrature_na():
    X_els = mesh.coords[mesh.conns, :]
    U_els = U[mesh.conns, :]
    dispGrads = vmap(fs_2_na.compute_field_gradient, in_axes=(0, 0))(U_els, X_els)
    nElements = mesh.num_elements
    npts = quadratureRule_1.xigauss.shape[0]
    exact = np.tile(targetDispGrad, (nElements, npts, 1, 1))
    assert np.allclose(dispGrads, exact)


def test_integrate_constant_field_multi_point_point_quadrature():
    integralOfOne = integrate_over_block(fs_2,
                                         U,
                                         mesh.coords,
                                         state_2,
                                         props,
                                         dt,
                                         lambda u, gradu, state, props, X, dt: 1.0,
                                         mesh.blocks['block'])
    np.isclose(integralOfOne, 1.0)


def test_integrate_constant_field_multi_point_quadrature_na():
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]
    integralOfOne = fs_2_na.integrate_on_elements(
        U_els,
        X_els,
        state_2,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    np.isclose(integralOfOne, 1.0)



# TODO add integration test/method for new na fspace


def test_integrate_linear_field_multi_point_quadrature():
    Ix = integrate_over_block(fs_2,
                              U,
                              mesh.coords,
                              state_2,
                              props,
                              dt,
                              lambda u, gradu, state, props, X, dt: gradu[0,0],
                              mesh.blocks['block'])
    idx = np.argmax(mesh.coords[:,0])
    expected = U[idx,0]*(yRange[1] - yRange[0])
    np.isclose(Ix, expected)
    
    Iy = integrate_over_block(fs_2,
                              U,
                              mesh.coords,
                              state_2,
                              props,
                              dt,
                              lambda u, gradu, state, props, X, dt: gradu[1,1],
                              mesh.blocks['block'])
    idx = np.argmax(mesh.coords[:,1])
    expected = U[idx,1]*(xRange[1] - xRange[0])
    np.isclose(Iy, expected)


def test_integrate_linear_field_multi_point_quadrature_na():
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]

    Ix = fs_2_na.integrate_on_elements(
        U_els,
        X_els,
        state_2,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[0, 0],
    )
    idx = np.argmax(mesh.coords[:,0])
    expected = U[idx,0]*(yRange[1] - yRange[0])
    np.isclose(Ix, expected)

    Iy = fs_2_na.integrate_on_elements(
        U_els,
        X_els,
        state_2,
        props,
        dt,
        lambda u, gradu, state, props, X, dt: gradu[1, 1],
    )

    idx = np.argmax(mesh.coords[:,1])
    expected = U[idx,1]*(xRange[1] - xRange[0])
    np.isclose(Iy, expected)


def test_integrate_over_half_block():
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0
    
    blockWithHalfTheVolume = slice(0,nElements//2)
    integral = integrate_over_block(fs_2,
                                    U,
                                    mesh.coords,
                                    state_2,
                                    props,
                                    dt,
                                    lambda u, gradu, state, props, X, dt: 1.0,
                                    blockWithHalfTheVolume)
    np.isclose(integral, 1.0/2.0)


def test_integrate_over_half_block_na():
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0
    blockWithHalfTheVolume = slice(0,nElements//2)

    U_els = U[mesh.conns[blockWithHalfTheVolume], :]
    X_els = U[mesh.conns[blockWithHalfTheVolume], :]

    integral = fs_2_na.integrate_on_elements(
        U_els,
        X_els,
        state_2[blockWithHalfTheVolume, :, :],
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    np.isclose(integral, 1.0/2.0)


def test_integrate_over_half_block_indices():
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0
    
    blockWithHalfTheVolume = np.arange(nElements//2)
    
    integral = integrate_over_block(fs_2,
                                    U,
                                    mesh.coords,
                                    state_2,
                                    props,
                                    dt,
                                    lambda u, gradu, state, props, X, dt: 1.0,
                                    blockWithHalfTheVolume)
    np.isclose(integral, 1.0/2.0)
        

def test_integrate_over_half_block_indices_na():
    nElements = mesh.num_elements
    # this test will only work with an even number of elements
    # put this in so that if test is modified to odd number,
    # we understand why it fails
    assert nElements % 2 == 0
    
    blockWithHalfTheVolume = np.arange(nElements//2)

    U_els = U[mesh.conns[blockWithHalfTheVolume], :]
    X_els = U[mesh.conns[blockWithHalfTheVolume], :]

    integral = fs_2_na.integrate_on_elements(
        U_els,
        X_els,
        state_2[blockWithHalfTheVolume, :, :],
        props,
        dt,
        lambda u, gradu, state, props, X, dt: 1.0,
    )
    np.isclose(integral, 1.0/2.0)


def test_jit_on_integration():
    integrate_jit = jax.jit(integrate_over_block, static_argnums=(6,))
    I = integrate_jit(fs_2, U, mesh.coords, state_2, props, dt, lambda u, gradu, state, props, X, dt: 1.0, mesh.blocks['block'])
    np.isclose(I, 1.0)


def test_jit_on_integration_na():
    integrate_jit = eqx.filter_jit(fs_2_na.integrate_on_elements)
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]
    I = integrate_jit(U_els, X_els, state_2, props, dt, lambda u, gradu, state, props, X, dt: 1.0)
    np.isclose(I, 1.0)


def test_jit_and_jacrev_on_integration():
    F = jax.jit(jax.jacrev(integrate_over_block, 1), static_argnums=(6,))
    dI = F(fs_2, U, mesh.coords, state_2, props, dt, lambda u, gradu, state, props, X, dt: 0.5*np.tensordot(gradu, gradu), mesh.blocks['block'])
    nNodes = mesh.coords.shape[0]
    interiorNodeIds = np.setdiff1d(np.arange(nNodes), mesh.nodeSets['all_boundary'])
    np.array_equal(dI[interiorNodeIds,:], np.zeros_like(U[interiorNodeIds,:]))


def test_jit_and_jacrev_on_integration_na():
    F = eqx.filter_jit(eqx.filter_jacrev(fs_2_na.integrate_on_elements))
    U_els = U[mesh.conns[mesh.blocks['block']], :]
    X_els = U[mesh.conns[mesh.blocks['block']], :]
    dI = F(U_els, X_els, state_2, props, dt, lambda u, gradu, state, props, X, dt: 0.5*np.tensordot(gradu, gradu))
    nNodes = mesh.coords.shape[0]
    interiorNodeIds = np.setdiff1d(np.arange(nNodes), mesh.nodeSets['all_boundary'])
    np.array_equal(dI[interiorNodeIds,:], np.zeros_like(U[interiorNodeIds,:]))
