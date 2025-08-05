from pancax import fem
from pancax import create_nodesets_from_sidesets
from .utils import create_mesh_and_disp
import jax.numpy as np
import numpy as onp

Nx = 3
Ny = 2
xRange = [0.0, 1.0]
yRange = [0.0, 1.0]
targetDispGrad = np.array([[0.1, -0.2], [0.4, -0.1]])


def triangle_inradius(tcoords):
    tcoords = onp.hstack((tcoords, onp.ones((tcoords.shape[0], 1))))

    area = 0.5 * onp.cross(tcoords[1] - tcoords[0], tcoords[2] - tcoords[0])[2]
    peri = (
        onp.linalg.norm(tcoords[1] - tcoords[0])
        + onp.linalg.norm(tcoords[2] - tcoords[1])
        + onp.linalg.norm(tcoords[0] - tcoords[2])
    )
    return area / peri


mesh, U = create_mesh_and_disp(
    Nx, Ny, xRange, yRange, lambda x: np.dot(targetDispGrad, x)
)


def test_create_nodesets_from_sidesets():
    # mesh, U = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: np.dot(targetDispGrad, x))
    nodeSets = create_nodesets_from_sidesets(mesh)

    # this test relies on the fact that matching nodesets
    # and sidesets were created on the MeshFixture

    for key in mesh.sideSets:
        assert np.array_equal(mesh.nodeSets[key], nodeSets[key])


def test_edge_connectivities():
    edgeConns, _ = fem.create_edges(mesh.conns)

    goldBoundaryEdgeConns = np.array([[0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 0]])

    # Check that every boundary edge has been found.
    # Boundary edges must appear with the same connectivity order,
    # since by convention boundary edge connectivities go
    # in the counter-clockwise sense.

    nBoundaryEdges = goldBoundaryEdgeConns.shape[0]
    boundaryEdgeFound = onp.full(nBoundaryEdges, False)

    for i, be in enumerate(goldBoundaryEdgeConns):
        rowsMatchingGold = onp.all(edgeConns == be, axis=1)
        boundaryEdgeFound[i] = onp.any(rowsMatchingGold)

    assert onp.all(boundaryEdgeFound)

    # Check that every interior edge as been found.
    # Interior edges have no convention defining which
    # sense the vertices should be ordered, so we check
    # for both permutations.

    goldInteriorEdgeConns = np.array([[0, 4], [1, 4], [1, 5]])

    nInteriorEdges = goldInteriorEdgeConns.shape[0]
    interiorEdgeFound = onp.full(nInteriorEdges, False)
    for i, e in enumerate(goldInteriorEdgeConns):
        foundWithSameSense = onp.any(onp.all(edgeConns == e, axis=1))
        foundWithOppositeSense = onp.any(onp.all(edgeConns == onp.flip(e), axis=1))
        interiorEdgeFound[i] = foundWithSameSense or foundWithOppositeSense

    assert onp.all(interiorEdgeFound)


def test_edge_to_neighbor_cells_data():
    edgeConns, edges = fem.create_edges(mesh.conns)

    goldBoundaryEdgeConns = np.array([[0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 0]])

    goldBoundaryEdges = onp.array(
        [
            [0, 0, -1, -1],
            [2, 0, -1, -1],
            [2, 1, -1, -1],
            [3, 1, -1, -1],
            [1, 1, -1, -1],
            [1, 2, -1, -1],
        ]
    )

    for be, bc in zip(goldBoundaryEdges, goldBoundaryEdgeConns):
        i = np.where(onp.all(edgeConns == bc, axis=1))
        assert np.all(edges[i, :] == be)

    goldInteriorEdgeConns = np.array([[0, 4], [1, 4], [5, 1]])
    goldInteriorEdges = onp.array([[1, 0, 0, 2], [0, 1, 3, 2], [2, 2, 3, 0]])

    for ie, ic in zip(goldInteriorEdges, goldInteriorEdgeConns):
        foundWithSameSense = onp.any(onp.all(edgeConns == ic, axis=1))
        foundWithOppositeSense = onp.any(onp.all(edgeConns == onp.flip(ic), axis=1))
        edgeDataMatches = False
        if foundWithSameSense:
            i = onp.where(onp.all(edgeConns == ic, axis=1))
            edgeData = ie
        elif foundWithOppositeSense:
            i = onp.where((onp.all(edgeConns == onp.flip(ic), axis=1)))
            edgeData = ie[[2, 3, 0, 1]]
        else:
            # self.fail('edge not found with vertices ' + str(ic))
            print("Need to raise an exception test here")
        edgeDataMatches = np.all(edges[i, :] == edgeData)
        assert edgeDataMatches


def test_conversion_to_quadratic_mesh_is_valid():
    newMesh = fem.mesh.create_higher_order_mesh_from_simplex_mesh(mesh, 2)

    nNodes = newMesh.coords.shape[0]
    assert nNodes == 15

    # make sure all of the newly created nodes got used in the connectivity
    assert np.array_equal(np.unique(newMesh.conns.ravel()), np.arange(nNodes))

    # check that all triangles are valid:
    # compute inradius of each triangle and of the sub-triangle of the mid-edge nodes
    # Both should be nonzero, and parent inradius should be 2x sub-triangle inradius
    master = newMesh.parentElement
    for t in newMesh.conns:
        elCoords = newMesh.coords[t, :]
        parentCoords = elCoords[master.vertexNodes, :]
        midEdgeNodes = master.faceNodes[:, 1]
        childCoords = elCoords[midEdgeNodes, :]

        parentArea = triangle_inradius(parentCoords)
        childArea = triangle_inradius(childCoords)

        assert parentArea > 0.0
        assert childArea > 0.0
        assert np.abs(parentArea - 2.0 * childArea) < 1e-10
