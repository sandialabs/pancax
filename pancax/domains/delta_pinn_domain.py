from jax import vmap
from jaxtyping import Array, Float, Int
from pancax.bcs import EssentialBC, NaturalBC
from pancax.fem import DofManager
from pancax.fem import FunctionSpace
from pancax.fem import Mesh
from pancax.timer import Timer
from pancax.kernels import LaplaceBeltrami, PhysicsKernel
from pancax.networks import FixedProperties
from pancax.physics import mass_matrix, stiffness_matrix
from pancax.post_processor import PostProcessor
from .variational_domain import VariationalDomain
from typing import List, Optional, Union
import jax.numpy as jnp
import numpy as onp
from scipy.sparse import linalg


class DeltaPINNDomain(VariationalDomain):
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
  :param n_eigen_values: Number of eigenvalues to use
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
  n_eigen_values: int
  eigen_modes: Float[Array, "nn nev"]

  def __init__(
    self,
    physics: PhysicsKernel,
    essential_bcs: List[EssentialBC],
    natural_bcs: List[NaturalBC],
    mesh_file: str,
    times: Float[Array, "nt"],
    n_eigen_values: int,
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
    :param n_eigen_values: Number of eigenvalues to use 
    """
    if not physics.use_delta_pinn:
      raise ValueError('Need a physics object set up for DeltaPINN')

    with Timer('DeltaPINNDomain.__init__'):
      super().__init__(
        physics, essential_bcs, natural_bcs, mesh_file, times, 
        p_order=p_order, q_order=q_order
      )
      self.n_eigen_values = n_eigen_values    
      self.eigen_modes = self.solve_eigen_problem(mesh_file, p_order, q_order)

  def field_values(self, field_network, t):
    us = vmap(self.physics.field_values, in_axes=(None, 0, None, 0))(
      field_network, self.coords, t, self.eigen_modes
    )
    return us

  def solve_eigen_problem(self, mesh_file, p_order, q_order):
      with Timer('DeltaPINNDomain.solve_eigen_problem'):
        times = jnp.zeros(1)
        bc_func = lambda x, t, nn: nn
        physics = LaplaceBeltrami(mesh_file, bc_func, self.n_eigen_values)
        domain = VariationalDomain(
          physics, [], [], mesh_file, times, 
          p_order=p_order, q_order=q_order
        )
        props = FixedProperties([])
        Uu = jnp.zeros(domain.dof_manager.get_unknown_size())
        U = domain.dof_manager.create_field(Uu)
        K = stiffness_matrix(domain, U, props)
        M = mass_matrix(domain, U, props)
            
        with Timer('eigen solve'):
          nModes = self.n_eigen_values
          mu, modes = linalg.eigsh(
            A=K, k=nModes, M=M, which='SM'
          )

          lambdas = 1. / mu
          for n in range(len(lambdas)):
            print(f'  Eigen mode {n} = {1. / lambdas[n]}')

        with Timer('post-processing'):
          pp = PostProcessor(mesh_file)
          pp.init(domain, 'output-eigen.e',
            node_variables=[
              'eigenvector'
            ]        
          )
          for n in range(len(lambdas)):
            print(f'  Post-processing eigenvector {n}')
            field = domain.dof_manager.create_field(modes[:, n])
            field = onp.asarray(field)
            pp.exo.put_time(n + 1, 1. / lambdas[n])
            pp.exo.put_node_variable_values('eigenvector', n + 1, field[:, 0])
          pp.close()

      # normalizing modes by default
      modes = jnp.array(modes[:, 0:len(lambdas) - 1])
      print(modes.shape)
      # modes = (modes - jnp.min(modes, axis=0)) / \
      #         (jnp.max(modes, axis=0) - jnp.min(modes, axis=0))

      return modes
