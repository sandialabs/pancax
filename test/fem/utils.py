from jax import vmap
from pancax.fem import construct_mesh_from_basic_data, create_structured_mesh_data
from pancax.fem.surface import create_edges
import jax.numpy as jnp


def create_mesh_and_disp(Nx, Ny, xRange, yRange, initial_disp_func, setNamePostFix=""):
    coords, conns = create_structured_mesh_data(Nx, Ny, xRange, yRange)
    tol = 1e-7
    nodeSets = {}
    nodeSets["left" + setNamePostFix] = jnp.flatnonzero(coords[:, 0] < xRange[0] + tol)
    nodeSets["bottom" + setNamePostFix] = jnp.flatnonzero(
        coords[:, 1] < yRange[0] + tol
    )
    nodeSets["right" + setNamePostFix] = jnp.flatnonzero(coords[:, 0] > xRange[1] - tol)
    nodeSets["top" + setNamePostFix] = jnp.flatnonzero(coords[:, 1] > yRange[1] - tol)
    nodeSets["all_boundary" + setNamePostFix] = jnp.flatnonzero(
        (coords[:, 0] < xRange[0] + tol)
        | (coords[:, 1] < yRange[0] + tol)
        | (coords[:, 0] > xRange[1] - tol)
        | (coords[:, 1] > yRange[1] - tol)
    )

    def is_edge_on_left(xyOnEdge):
        return jnp.all(xyOnEdge[:, 0] < xRange[0] + tol)

    def is_edge_on_bottom(xyOnEdge):
        return jnp.all(xyOnEdge[:, 1] < yRange[0] + tol)

    def is_edge_on_right(xyOnEdge):
        return jnp.all(xyOnEdge[:, 0] > xRange[1] - tol)

    def is_edge_on_top(xyOnEdge):
        return jnp.all(xyOnEdge[:, 1] > yRange[1] - tol)

    sideSets = {}
    sideSets["left" + setNamePostFix] = create_edges(coords, conns, is_edge_on_left)
    sideSets["bottom" + setNamePostFix] = create_edges(coords, conns, is_edge_on_bottom)
    sideSets["right" + setNamePostFix] = create_edges(coords, conns, is_edge_on_right)
    sideSets["top" + setNamePostFix] = create_edges(coords, conns, is_edge_on_top)
    print(sideSets.values())
    allBoundaryEdges = jnp.vstack([s for s in sideSets.values()])
    sideSets["all_boundary" + setNamePostFix] = allBoundaryEdges

    blocks = {"block" + setNamePostFix: jnp.arange(conns.shape[0])}
    mesh = construct_mesh_from_basic_data(coords, conns, blocks, nodeSets, sideSets)
    print(mesh)
    return mesh, vmap(initial_disp_func)(mesh.coords)
