from .forward_problem import ForwardProblem
from pancax.bcs import EssentialBC, NaturalBC
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
    essential_bcs: List[EssentialBC],
    natural_bcs: List[NaturalBC],
    field_data: FullFieldData,
    global_data: GlobalData
  ) -> None:
    super().__init__(domain, physics, ics, essential_bcs, natural_bcs)
    self.field_data = field_data
    self.global_data = global_data
