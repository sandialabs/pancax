from jax import vmap
from pancax.bcs import EssentialBC
from pancax.fem import QuadratureRule
from pancax.fem.dof_manager import DofManager
from pancax.fem.function_space import construct_function_space
from pancax.fem.mesh import construct_mesh_from_basic_data
from pancax.fem.mesh import create_structured_mesh_data
from pancax.fem.surface import create_edges
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


Nx = 4
Ny = 5
xRange = [0., 1.]
yRange = [0., 1.]

mesh, _ = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x : 0*x)

quadRule = QuadratureRule(mesh.parentElement, 1)
fs = construct_function_space(mesh, quadRule)
ebcs = [EssentialBC(nodeSet='top', component=0),
        EssentialBC(nodeSet='right', component=1)]
nNodes = Nx * Ny
nFields = 2
dofManager = DofManager(mesh, nFields, ebcs)

nDof = nFields * nNodes
U = np.zeros((nNodes, nFields))
U = U.at[:,1].set(1.0)
U = U.at[mesh.nodeSets['top'],0].set(2.0)
U = U.at[mesh.nodeSets['right'],1].set(3.0)

 
def test_get_bc_size():
    # number of dofs from top, field 0
    nEbcs = Nx
    # number of dofs from right, field 1
    nEbcs += Ny
    assert dofManager.get_bc_size() == nEbcs


def test_get_unknown_size():
    # number of dofs from top, field 0
    nEbcs = Nx
    # number of dofs from right, field 1
    nEbcs += Ny
    assert dofManager.get_unknown_size() == nDof - nEbcs


def test_slice_unknowns_with_dof_indices():
    Uu = dofManager.get_unknown_values(U)
    Uu_x = dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None),0) )
    np.array_equal(Uu_x, np.zeros(Nx*(Ny-1)))
    Uu_y = dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None),1) )
    np.array_equal(Uu_y, np.ones(Ny*(Nx-1)))


def test_create_field_and_get_bc_values():
    Uu = np.zeros(dofManager.get_unknown_size())
    U_new = dofManager.create_field(Uu, Ubc=5.0)
    assert U_new.shape == U.shape
    assert np.allclose(dofManager.get_bc_values(U_new), 5.0)

