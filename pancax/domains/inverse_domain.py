from .variational_domain import VariationalDomain
from jaxtyping import Array, Float
from pancax.bcs import EssentialBC
from pancax.data import FullFieldData, GlobalData
from pancax.fem import DofManager
from pancax.fem import FunctionSpace
from pancax.kernels import PhysicsKernel
from typing import List, Optional, Union


class InverseDomain(VariationalDomain):
  """
  Inverse domain type derived from a ForwardDomain

  Note that currently this likely only supports single block meshes.

  :param physics: A physics kernel to use for physics calculations.
  :param dof_manager: A DofManager to track what dofs are free/fixed.
  :param fspace: A FunctionSpace to help with integration
  :param q_rule_1d: A quadrature rule for line/surface integrations. TODO rename this
  :param q_rule_2d: A quadrature rule for cell integrations. TODO rename this
  :param coords: Nodal coordinates in the reference configuration.
  :param conns: Element connectivity matrix
  :param field_data: Data structure that holds the full field data.
  :param global_data: Data structure that holds the global data.
  """
  physics: PhysicsKernel
  dof_manager: DofManager
  fspace: FunctionSpace
  fspace_centroid: FunctionSpace
  coords: Float[Array, "nn nd"]
  conns: Float[Array, "ne nnpe"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
  field_data: FullFieldData
  global_data: GlobalData

  def __init__(
    self,
    physics: PhysicsKernel,
    essential_bcs: List[EssentialBC],
    natural_bcs: any, # TODO figure out to handle this
    mesh_file: str,
    times: Float[Array, "nt"],
    field_data: FullFieldData,
    global_data: GlobalData,
    p_order: Optional[int] = 1,
    q_order: Optional[int] = 2
  ) -> None:
    """
    :param physics: A ``PhysicsKernel`` object
    :param essential_bcs: A list of ``EssentiablBC`` objects
    :param natural_bcs: TODO
    :param mesh_file: mesh file name as string
    :param times: An array of time values to use
    :param field_data: ``FieldData`` object
    :param global_data: ``GlobalData`` object
    :param p_order: Polynomial order for mesh. Only hooked up to tri meshes.
    :param q_order: Quadrature order to use. 
    """
    super().__init__(
      physics, essential_bcs, natural_bcs, mesh_file, times,
      p_order=p_order, q_order=q_order
    )
    self.field_data = field_data
    self.global_data = global_data
