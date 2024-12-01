from jaxtyping import Array, Float
from pancax import fem
from pancax.fem import surface
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp


BCFunc = Callable[[Float[Array, "nd"], float], Float[Array, "nf"]]
# remove component from the definition.
# it doesn't appear to be doing anything
# class NaturalBC(NamedTuple):
class NaturalBC(eqx.Module):
  sideset: str
  function: Optional[BCFunc] = lambda x, t: 0.0

  def coordinates(self, mesh, q_rule_1d):
    xigauss, wgauss = q_rule_1d
    edges = mesh.sideSets[self.sideset]

    def vmap_inner(edge):
      edge_coords = surface.get_coords(mesh.coords, mesh.conns, edge)
      # edge_coords = fem.mesh.get.get_coords(mesh, side)
      jac = jnp.linalg.norm(edge_coords[0,:] - edge_coords[1,:])
      xgauss = edge_coords[0] + jnp.outer(xigauss, edge_coords[1] - edge_coords[0])
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
