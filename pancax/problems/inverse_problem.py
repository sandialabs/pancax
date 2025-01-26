from .forward_problem import ForwardProblem
from pancax.bcs import DirichletBC, NeumannBC
from pancax.data import FullFieldData, GlobalData
from pancax.domains import BaseDomain
from pancax.physics_kernels import BasePhysics
from typing import Callable, List


class InverseProblem(ForwardProblem):
  field_data: FullFieldData
  global_data: GlobalData

  def __init__(
    self,
    domain: BaseDomain,
    physics: BasePhysics,
    ics: List[Callable],
    dirichlet_bcs: List[DirichletBC],
    neumann_bcs: List[NeumannBC],
    field_data: FullFieldData,
    global_data: GlobalData
  ) -> None:
    super().__init__(domain, physics, ics, dirichlet_bcs, neumann_bcs)
    self.field_data = field_data
    self.global_data = global_data
