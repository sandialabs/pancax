from .base import AbstractPancaxModel
from .fields import Field
from .mlp import MLP
from jaxtyping import Array, Float
from typing import Callable, Optional, Union
import equinox as eqx
import jax.tree_util as jtu


State = Union[Float[Array, "nt ne nq ns"], eqx.Module, None]


class _Parameters(AbstractPancaxModel):
    """
    Data structure for storing all parameters
    needed for a model

    :param fields: field network parameters object
    :param physics: physics object
    :param state: state object (can be parameters or jax array)
    """

    fields: Field
    physics: eqx.Module
    state: State

    def __init__(
        self,
        problem,
        constitutive_model,
        key,
        dirichlet_bc_func: Optional[Callable] = lambda x, t, z: z,
        network_type: Optional[eqx.Module] = MLP,
        seperate_networks: Optional[bool] = False
    ) -> None:
        fields = Field(
            problem, key,
            dirichlet_bc_func=dirichlet_bc_func,
            network_type=network_type,
            seperate_networks=seperate_networks
        )
        # TODO what do we do with this guy?
        state = None

        physics = eqx.tree_at(
            lambda x: x.constitutive_model, problem.physics, constitutive_model
        )
        self.fields = fields
        self.physics = physics
        self.state = state

    def __iter__(self):
        return iter((self.fields, self.physics, self.state))


class Parameters(AbstractPancaxModel):
    is_ensemble: bool = eqx.field(static=True)
    n_ensemble: int = eqx.field(static=True)
    parameters: _Parameters

    def __init__(
        self,
        problem,
        key,
        dirichlet_bc_func: Optional[Callable] = lambda x, t, z: z,
        network_type: Optional[eqx.Module] = MLP,
        seperate_networks: Optional[bool] = False
    ) -> None:
        if len(key.shape) == 1:
            is_ensemble = False
            n_ensemble = 1
            parameters = _Parameters(
                problem, problem.physics.constitutive_model, key,
                dirichlet_bc_func=dirichlet_bc_func,
                network_type=network_type,
                seperate_networks=seperate_networks
            )
        elif len(key.shape) == 2:
            is_ensemble = True
            n_ensemble = key.shape[0]

            @eqx.filter_vmap
            def vmap_func(key, constitutive_model):
                return _Parameters(
                    problem, constitutive_model, key,
                    dirichlet_bc_func=dirichlet_bc_func,
                    network_type=network_type,
                    seperate_networks=seperate_networks
                )

            parameters = vmap_func(key, problem.physics.constitutive_model)
        else:
            raise ValueError(
                f"Invalid shape for key {key} with shape {key.shape}"
            )

        self.is_ensemble = is_ensemble
        self.n_ensemble = n_ensemble
        self.parameters = parameters

    def __iter__(self):
        return iter((self.fields, self.physics, self.state))

    @property
    def fields(self):
        return self.parameters.fields

    @property
    def physics(self):
        return self.parameters.physics

    @property
    def state(self):
        return self.parameters.state

    def freeze_fields_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        fields_filter = jtu.tree_map(lambda _: False, self.parameters.fields)
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.fields, filter_spec, fields_filter
        )

        # freeze normalization
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.fields.x_mins, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.fields.x_maxs, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.fields.t_min, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.fields.t_max, filter_spec, replace=False
        )
        return filter_spec

    # Move some of below to actually network implementation
    def freeze_physics_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        physics_filter = jtu.tree_map(lambda _: False, self.parameters.physics)
        filter_spec = eqx.tree_at(
            lambda x: x.parameters.physics, filter_spec, physics_filter
        )
        return filter_spec

    def freeze_physics_normalization_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        filter_spec = eqx.tree_at(
            lambda tree: tree.parameters.fields.x_mins, filter_spec,
            replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.parameters.fields.x_maxs, filter_spec,
            replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.parameters.fields.t_min, filter_spec,
            replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.parameters.fields.t_max, filter_spec,
            replace=False
        )
        return filter_spec
