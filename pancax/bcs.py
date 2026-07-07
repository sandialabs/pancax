from jaxtyping import Array, Float
from typing import Callable, List, Optional, Union
import equinox as eqx


BCFunc = Callable[[Float[Array, "nd"], float], float]
SetName = Union[None, str]

# TODO build a type hiearchy
# class BaseBC(eqx.Module):
#     function: BCFunc

class BCFunction(eqx.Module):
    func: BCFunc

    def __call__(self, x, t):
        return self.bc_func(x, t)


class DirichletBC(eqx.Module):
    # function: BCFunc
    function: BCFunction
    block_name: SetName
    nset_name: SetName
    sset_name: SetName
    component: int

    def __init__(
        self,
        component,
        function: Optional[BCFunc] = lambda x, t: 0.0,
        block_name: Optional[SetName] = None,
        nset_name: Optional[SetName] = None,
        sset_name: Optional[SetName] = None
    ) -> None:
        self.component = component
        self.function = BCFunction(function)
        self.block_name = block_name
        self.nset_name = nset_name
        self.sset_name = sset_name

    def coordinates(self, mesh):
        nodes = mesh.nodeSets[self.nset_name]
        coords = mesh.coords
        return coords[nodes, :]


class NeumannBC(eqx.Module):
    sideset: str
    function: Optional[BCFunc] = lambda x, t: 0.0

    def coordinates(self, mesh, q_rule_1d):
        xigauss, wgauss = q_rule_1d
        edges = mesh.sideSets[self.sideset]

        def vmap_inner(edge):
            edge_coords = surface.get_coords(mesh.coords, mesh.conns, edge)
            # edge_coords = fem.mesh.get.get_coords(mesh, side)
            # jac = jnp.linalg.norm(edge_coords[0, :] - edge_coords[1, :])
            xgauss = edge_coords[0] + jnp.outer(
                xigauss, edge_coords[1] - edge_coords[0]
            )
            return xgauss

        edge_coords = jax.vmap(vmap_inner)(edges)
        edge_coords = jnp.vstack(edge_coords)
        return edge_coords

    def normals(self, mesh, q_rule_1d):
        xigauss, wgauss = q_rule_1d
        edges = mesh.sideSets[self.sideset]
        n_gauss_points = xigauss.shape[0]

        def vmap_inner(edge):
            edge_coords = surface.get_coords(mesh.coords, mesh.conns, edge)
            normal = surface.compute_normal(edge_coords)
            normals = jnp.tile(normal, (n_gauss_points, 1))
            return normals

        edge_normals = jax.vmap(vmap_inner)(edges)
        edge_normals = jnp.vstack(edge_normals)
        return edge_normals
