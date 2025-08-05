from .base import AbstractPancaxModel
from .fields import Field
from .mlp import MLP
from jaxtyping import Array, Float
from typing import Optional, Union
import equinox as eqx
import jax.tree_util as jtu


State = Union[Float[Array, "nt ne nq ns"], eqx.Module, None]


class Parameters(AbstractPancaxModel):
    """
    Data structure for storing all parameters
    needed for a model

    :param fields: field network parameters object
    :param physics: physics object
    :param state: state object (can be parameters or jax array)
    """

    fields: eqx.Module
    physics: eqx.Module
    state: State

    def __init__(
        self,
        problem,
        key,
        network_type: Optional[eqx.Module] = MLP,
        seperate_networks: Optional[bool] = False
    ) -> None:
        fields = Field(
            problem, key,
            network_type=network_type,
            seperate_networks=seperate_networks
        )
        # state = self._create_state_array(problem)
        state = None

        self.fields = fields
        self.physics = problem.physics
        self.state = state

    def __iter__(self):
        return iter((self.fields, self.physics, self.state))

    # TODO
    # make helper filter methods so there's
    # less code duplication
    def freeze_fields_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        fields_filter = jtu.tree_map(lambda _: False, self.fields)
        filter_spec = eqx.tree_at(
            lambda x: x.fields, filter_spec, fields_filter
        )

        # freeze normalization
        filter_spec = eqx.tree_at(
            lambda x: x.physics.x_mins, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.physics.x_maxs, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.physics.t_min, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda x: x.physics.t_max, filter_spec, replace=False
        )
        return filter_spec

    # Move some of below to actually network implementation
    def freeze_physics_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        physics_filter = jtu.tree_map(lambda _: False, self.physics)
        filter_spec = eqx.tree_at(
            lambda x: x.physics, filter_spec, physics_filter
        )
        return filter_spec

    def freeze_physics_normalization_filter(self):
        filter_spec = jtu.tree_map(lambda _: True, self)
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.x_mins, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.x_maxs, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.t_min, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.t_max, filter_spec, replace=False
        )
        return filter_spec
