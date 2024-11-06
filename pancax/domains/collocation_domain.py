from .base_domain import BaseDomain
from jaxtyping import Array, Float
from pancax.bcs import EssentialBC, NaturalBC
from pancax.fem import DofManager
from pancax.fem import Mesh
from pancax.fem import QuadratureRule
from pancax.fem.elements import LineElement
from pancax.kernels import PhysicsKernel
from pancax.timer import Timer
from typing import List, Optional, Union
import jax
import jax.numpy as jnp


class CollocationDomain(BaseDomain):
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
  mesh: Mesh
  coords: Float[Array, "nn nd"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
  physics: PhysicsKernel
  essential_bcs: List[EssentialBC]
  natural_bcs: List[NaturalBC]
  dof_manager: DofManager
  neumann_xs: Float[Array, "nn nd"]
  neumann_ns: Float[Array, "nn nd"]
  # neumann_outputs: Float[Array, "nn nf"]  

  def __init__(
    self,
    physics: PhysicsKernel,
    essential_bcs: List[EssentialBC],
    natural_bcs: List[NaturalBC],
    mesh_file: str,
    times: Float[Array, "nt"],
    p_order: Optional[int] = 1,
    q_order: Optional[int] = 2,
    vectorize_over_time: Optional[bool] = False
  ) -> None:
    """
    :param physics: A ``PhysicsKernel`` object
    :param essential_bcs: A list of ``EssentiablBC`` objects
    :param natural_bcs: TODO
    :param mesh_file: mesh file name as string
    :param times: set of times
    :param p_order: Polynomial order for mesh. Only hooked up to tri meshes.
    :param q_order: Quadrature order to use. 
    :param vectorize_over_time: Flag to enable vectorization over time
      this likely only makes sense for path-independent problems.
    """
    with Timer('CollocationDomain.__init__'):
      super().__init__(
        physics, essential_bcs, natural_bcs, 
        mesh_file, times, 
        p_order=p_order
      )

      # TODO currently all of the below is busted for transient problems

      # TODO need to gather dirichlet inputs/outputs
      # mesh = self.fspace.mesh
      # if len(essential_bcs) > 0:
      #   self.dirichlet_xs = jnp.vstack([bc.coordinates(mesh) for bc in essential_bcs])
      #   self.dirichlet_outputs = jnp.vstack([
      #     jax.vmap(bc.function, in_axes=(0, None))(bc.coordinates(mesh), )
      #   ])

      # TODO below should eventually be move to the base class maybe?
      # get neumann xs and ns
      # mesh = self.fspace.mesh
      mesh = self.mesh
      if mesh.num_dimensions != 2:
        raise ValueError(
          'Only 2D meshes currently supported for collocation problems. '
          'Need to implement surface normal calculations on 3D elements.'
        )
      # q_rule_1d = self.q_rule_1d
      # currently only support 2D meshes
      q_rule_1d = QuadratureRule(LineElement(1), q_order)
      if len(natural_bcs) > 0:
        self.neumann_xs = [bc.coordinates(mesh, q_rule_1d) for bc in natural_bcs]
        self.neumann_ns = [bc.normals(mesh, q_rule_1d) for bc in natural_bcs]
        print('Warning this neumann condition will fail for inhomogenous conditions with time')
        # self.neumann_outputs = jnp.vstack([
        #   jax.vmap(bc.function, in_axes=(0, None))(bc.coordinates(mesh, q_rule_1d), 0.0) \
        #   for bc in natural_bcs
        # ])
      else:
        self.neumann_xs = []
        self.neumann_ns = []
        # self.neumann_xs = jnp.array([[]])
        # self.neumann_ns = jnp.array([[]])
        # self.neumann_outputs = jnp.array([[]])
