from .forward_problem import ForwardProblem
from pancax.bcs import DirichletBC, NeumannBC
from pancax.data import FullFieldData, GlobalData
from pancax.domains import BaseDomain
from pancax.physics_kernels import BasePhysics
from typing import Callable, List, Optional


class InverseProblem(ForwardProblem):
    field_data: FullFieldData
    global_data: GlobalData

    def __init__(
        self,
        domain: BaseDomain,
        physics: BasePhysics,
        field_data: FullFieldData,
        global_data: GlobalData,
        ics: Optional[List[Callable]] = [],
        dirichlet_bcs: Optional[List[DirichletBC]] = [],
        neumann_bcs: Optional[List[NeumannBC]] = [],
    ) -> None:
        super().__init__(domain, physics, ics, dirichlet_bcs, neumann_bcs)
        self.field_data = field_data
        self.global_data = global_data
