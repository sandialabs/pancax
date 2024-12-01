from .base import BaseDomain
from jaxtyping import Array, Float, Int
from pancax.fem import DofManager, FunctionSpace, Mesh
from pancax.fem import NonAllocatedFunctionSpace, QuadratureRule
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp


class VariationalDomain(BaseDomain):
  mesh_file: str
  mesh: Mesh
  coords: Float[Array, "nn nd"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
  conns: Int[Array, "ne nnpe"]
  dof_manager: DofManager
  fspace: FunctionSpace
  fspace_centroid: FunctionSpace

  def __init__(
    self,
    mesh_file: str,
    times: Float[Array, "nt"],
    p_order: Optional[int] = 1,
    q_order: Optional[int] = 2
  ):
    super().__init__(mesh_file, times, p_order=p_order)
    self.conns = jnp.array(self.mesh.conns)
    self.dof_manager = None
    self.fspace = NonAllocatedFunctionSpace(
      self.mesh, QuadratureRule(self.mesh.parentElement, q_order)
    )
    self.fspace_centroid = NonAllocatedFunctionSpace(
      self.mesh, QuadratureRule(self.mesh.parentElement, 1)
    )

  def update_dof_manager(self, dirichlet_bcs, n_dofs):
    dof_manager = DofManager(self.mesh, n_dofs, dirichlet_bcs)
    dof_manager.isUnknown = jnp.array(dof_manager.isUnknown, dtype=jnp.bool)
    dof_manager.unknownIndices = jnp.array(dof_manager.unknownIndices)
    new_pytree = eqx.tree_at(lambda x: x.dof_manager, self, dof_manager)
    return new_pytree
