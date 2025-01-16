from ..bcs import EssentialBC, NaturalBC
from ..domains import BaseDomain, CollocationDomain, VariationalDomain
from ..physics_kernels import BasePhysics, BaseStrongFormPhysics, BaseVariationalFormPhysics
from typing import Callable, List
import equinox as eqx


class DomainPhysicsCompatabilityError(Exception):
  pass


# note that physics here will note hold the correct parameters
# after initialization
class ForwardProblem(eqx.Module):
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
    
    # check compatability between domain and physics
    if type(domain) == CollocationDomain:
      if not issubclass(type(physics), BaseStrongFormPhysics):
        raise DomainPhysicsCompatabilityError(
          f'Incompatable domain and physics.\n'
          f'Got domain of type = {type(domain)}\n'
          f'Got physics of type = {type(physics)}'
        )
    elif type(domain) == VariationalDomain:
      # TODO also need a weak form catch here
      # TODO or just maybe make a base variational physics class
      if not issubclass(type(physics), BaseVariationalFormPhysics):
        print(physics.__class__.__bases__)
        print(physics.__class__.__mro__)
        raise DomainPhysicsCompatabilityError(
          f'Incompatable domain and physics.\n'
          f'Got domain of type = {type(domain)}\n'
          f'Got physics of type = {type(physics)}'
        )
    else:
      assert False, 'wtf is this domain'

    if type(domain) == VariationalDomain:
      domain = domain.update_dof_manager(essential_bcs, physics.n_dofs)

    self.domain = domain
    physics = physics.update_normalization(domain)
    self.physics = physics.update_var_name_to_method()
    self.ics = ics
    self.essential_bcs = essential_bcs
    self.natural_bcs = natural_bcs

  # TODO a lot of these below are for some backwards
  # compatability during a transition period to
  # this new interface
  @property
  def conns(self):
    return self.domain.conns

  @property
  def coords(self):
    return self.domain.coords
  
  @property
  def dof_manager(self):
    return self.domain.dof_manager

  @property
  def fspace(self):
    return self.domain.fspace

  @property
  def mesh(self):
    return self.domain.mesh

  @property
  def mesh_file(self):
    return self.domain.mesh_file

  @property
  def times(self):
    return self.domain.times
