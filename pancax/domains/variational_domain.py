from .base_domain import BaseDomain
from jaxtyping import Array, Float, Int
from pancax.bcs import EssentialBC, NaturalBC
from pancax.fem import DofManager
from pancax.fem import FunctionSpace
from pancax.fem import Mesh
from pancax.fem import NonAllocatedFunctionSpace
from pancax.fem import QuadratureRule
from pancax.timer import Timer
from pancax.kernels import PhysicsKernel
from typing import List, Optional, Union
import jax.numpy as jnp


class VariationalDomain(BaseDomain):
  """
  Base domain for all problem types.
  This holds essential things for the problem
  such as a mesh to load a geometry from,
  times, physics, bcs, etc.

  :param mesh: A mesh from an exodus file most likely
  :param coords: An array of coordinates
  :param times: An array of times
  :param physics: An initialized physics object
  :param essential_bcs: A list of EssentialBCs
  :param natural_bcs: a list of NaturalBCs
  :param dof_manager: A DofManager for keeping track of essential bcs
  :param conns: An array of connectivities
  :param fspace: A FunctionSpace to help with integration
  :param fspace_centroid: A FunctionSpace to help with integration
  """
  mesh: Mesh
  coords: Float[Array, "nn nd"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
  physics: PhysicsKernel
  essential_bcs: List[EssentialBC]
  natural_bcs: List[NaturalBC]
  dof_manager: DofManager
  conns: Int[Array, "ne nnpe"]
  fspace: FunctionSpace
  fspace_centroid: FunctionSpace

  def __init__(
    self,
    physics: PhysicsKernel,
    essential_bcs: List[EssentialBC],
    natural_bcs: any, # TODO figure out to handle this
    mesh_file: str,
    times: Float[Array, "nt"],
    p_order: Optional[int] = 1,
    q_order: Optional[int] = 2
  ) -> None:
    """
    :param physics: A ``PhysicsKernel`` object
    :param essential_bcs: A list of ``EssentiablBC`` objects
    :param natural_bcs: TODO
    :param mesh_file: mesh file name as string
    :param times: set of times
    :param p_order: Polynomial order for mesh. Only hooked up to tri meshes.
    :param q_order: Quadrature order to use. 
    """
    with Timer('VariationalDomain.__init__'):
      super().__init__(
        physics, essential_bcs, natural_bcs, mesh_file, times, 
        p_order=p_order
      )
      with Timer('move connectivity to device'):
        self.conns = jnp.array(self.mesh.conns)
      
      self.fspace = NonAllocatedFunctionSpace(
        self.mesh, QuadratureRule(self.mesh.parentElement, q_order)
      )
      self.fspace_centroid = NonAllocatedFunctionSpace(
        self.mesh, QuadratureRule(self.mesh.parentElement, 1)
      )
