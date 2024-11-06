from jax import vmap
from jaxtyping import Array, Float
from pancax.bcs import EssentialBC, NaturalBC
from pancax.bcs import EssentialBCSet
from pancax.fem import DofManager, Mesh, SimplexTriElement
from pancax.fem import create_higher_order_mesh_from_simplex_mesh
from pancax.fem import read_exodus_mesh
from pancax.kernels import PhysicsKernel
from pancax.timer import Timer
from pancax.utils import find_mesh_file
from typing import List, Optional, Union
import equinox as eqx
import jax.numpy as jnp



class BaseDomain(eqx.Module):
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
  """
  mesh_file: str
  mesh: Mesh
  coords: Float[Array, "nn nd"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
  physics: PhysicsKernel
  essential_bcs: List[EssentialBC]
  natural_bcs: List[NaturalBC]
  dof_manager: DofManager

  def __init__(
    self,
    physics: PhysicsKernel,
    essential_bcs: List[EssentialBC],
    natural_bcs: any, # TODO figure out to handle this
    mesh_file: str,
    times: Float[Array, "nt"],
    p_order: Optional[int] = 1
  ):
    with Timer('BaseDomain.__init__'):
      # setup
      n_dofs = physics.n_dofs
      
      # mesh
      mesh = read_exodus_mesh(mesh_file)

      # if tri mesh, we can make it higher order from lower order
      if type(mesh.parentElement) == SimplexTriElement:
        mesh = create_higher_order_mesh_from_simplex_mesh(mesh, p_order, copyNodeSets=True)
      else:
        print('WARNING: Ignoring polynomial order flag for non tri mesh')

      with Timer('move coordinates to device'):
        coords = jnp.array(mesh.coords)

      # dof book keeping
      dof_manager = DofManager(mesh, n_dofs, essential_bcs)
      # TODO move below to dof manager
      dof_manager.isUnknown = jnp.array(dof_manager.isUnknown, dtype=jnp.bool)
      dof_manager.unknownIndices = jnp.array(dof_manager.unknownIndices)

      # setting all at once
      self.mesh_file = mesh_file
      self.mesh = mesh
      self.coords = coords
      self.times = times
      self.physics = physics
      # self.essential_bcs = essential_bcs
      self.essential_bcs = EssentialBCSet(essential_bcs)
      self.natural_bcs = natural_bcs
      self.dof_manager = dof_manager

  def field_values(self, field_network, t):
    us = vmap(self.physics.field_values, in_axes=(None, 0, None))(
      field_network, self.coords, t
    )
    return us