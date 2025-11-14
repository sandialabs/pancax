from ..bcs import DirichletBC, NeumannBC
from ..domains import (
    BaseDomain,
    CollocationDomain,
    DeltaPINNDomain,
    VariationalDomain
)
from ..physics_kernels import (
    BasePhysics,
    BaseStrongFormPhysics,
    BaseVariationalFormPhysics,
    BaseEnergyFormPhysics,
)
from typing import Callable, List, Optional
import equinox as eqx
import jax.numpy as jnp


class DomainPhysicsCompatabilityError(Exception):
    pass


# note that physics here will note hold the correct parameters
# after initialization
class ForwardProblem(eqx.Module):
    domain: BaseDomain
    physics: BasePhysics
    ics: List[Callable]
    dirichlet_bcs: List[DirichletBC]
    neumann_bcs: List[NeumannBC]
    is_delta_pinn: bool

    def __init__(
        self,
        domain: BaseDomain,
        physics: BasePhysics,
        ics: Optional[List[Callable]] = [],
        dirichlet_bcs: Optional[List[DirichletBC]] = [],
        neumann_bcs: Optional[List[NeumannBC]] = [],
    ) -> None:

        # check compatability between domain and physics
        if type(domain) is CollocationDomain:
            if not issubclass(type(physics), BaseStrongFormPhysics):
                raise DomainPhysicsCompatabilityError(
                    f"Incompatable domain and physics.\n"
                    f"Got domain of type = {type(domain)}\n"
                    f"Got physics of type = {type(physics)}"
                )
        elif type(domain) is VariationalDomain \
                or type(domain) is DeltaPINNDomain:
            # TODO also need a weak form catch here
            # TODO or just maybe make a base variational physics class
            if not issubclass(
                type(physics), BaseVariationalFormPhysics
            ) and not issubclass(type(physics), BaseEnergyFormPhysics):
                print(physics.__class__.__bases__)
                print(physics.__class__.__mro__)
                raise DomainPhysicsCompatabilityError(
                    f"Incompatable domain and physics.\n"
                    f"Got domain of type = {type(domain)}\n"
                    f"Got physics of type = {type(physics)}"
                )
        else:
            assert False, "wtf is this domain"

        if type(domain) is VariationalDomain or \
                type(domain) is DeltaPINNDomain:
            domain = domain.update_dof_manager(dirichlet_bcs, physics.n_dofs)

        # setup physics
        # physics = physics.update_normalization(domain)
        physics = physics.update_var_name_to_method()

        if type(domain) is DeltaPINNDomain:
            is_delta_pinn = True
            physics = eqx.tree_at(
                lambda x: x.x_mins, physics,
                jnp.min(domain.eigen_modes, axis=0)
            )
            physics = eqx.tree_at(
                lambda x: x.x_maxs, physics,
                jnp.max(domain.eigen_modes, axis=0)
            )
        else:
            is_delta_pinn = False

        self.domain = domain
        # physics = physics.update_normalization(domain)
        # self.physics = physics.update_var_name_to_method()
        self.physics = physics
        self.ics = ics
        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs
        self.is_delta_pinn = is_delta_pinn

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
    def n_dims(self):
        return self.domain.coords.shape[1]

    @property
    def n_dofs(self):
        return self.physics.n_dofs

    @property
    def times(self):
        return self.domain.times
