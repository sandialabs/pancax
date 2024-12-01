from .bcs import EssentialBC, NaturalBC
from .domains import BaseDomain, VariationalDomain
from .physics_kernels import BasePhysics
from typing import Callable, List
import equinox as eqx


# note that physics here will note hold the correct parameters
# after initialization
class Problem(eqx.Module):
  domain: BaseDomain
  physics: BasePhysics
  ics: List[Callable]
  essential_bcs: List[EssentialBC]
  natural_bcs: List[NaturalBC]

  def __init__(
    self, 
    domain: BaseDomain,
    physics: BasePhysics,
    ics: List[Callable],
    essential_bcs: List[EssentialBC],
    natural_bcs: List[NaturalBC]
  ) -> None:
    
    if type(domain) == VariationalDomain:
      domain = domain.update_dof_manager(essential_bcs, physics.n_dofs)

    self.domain = domain
    physics = physics.update_normalization(domain)
    self.physics = physics.update_var_name_to_method()
    self.ics = ics
    self.essential_bcs = essential_bcs
    self.natural_bcs = natural_bcs

  @property
  def coords(self):
    return self.domain.coords
  
  @property
  def mesh(self):
    return self.domain.mesh

  @property
  def times(self):
    return self.domain.times
