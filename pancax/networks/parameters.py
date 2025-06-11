from .base import BasePancaxModel
from .fields import Field
from .mlp import MLP
from ..domains import VariationalDomain
from jax import vmap
from jaxtyping import Array, Float
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu


State = Union[Float[Array, "nt ne nq ns"], eqx.Module, None]


class Parameters(BasePancaxModel):
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

    def _create_state_array(self, problem) -> State:
        if type(problem.domain) is VariationalDomain:
            nt = len(problem.domain.times)
            ne = problem.domain.conns.shape[0]
            nq = len(problem.domain.fspace.quadrature_rule)

            # TODO need a better interface
            if hasattr(problem.physics, 'constitutive_model'):
                if problem.physics.constitutive_model.num_state_variables > 0:
                    def _vmap_func(n):
                        return problem.physics.constitutive_model.\
                            initial_state()

                    state = vmap(vmap(vmap(_vmap_func)))(
                        jnp.zeros((nt, ne, nq))
                    )
                    print(state.shape)
                else:
                    state = jnp.zeros((nt, ne, nq, 0))
            return state
        else:
            print(
                "WARNING: Setting state to None since this is "
                "not a VariationalDomain"
            )
            return None

    # TODO
    # make helper filter methods so there's
    # less code duplication
    def freeze_fields_filter(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics, filter_spec, replace=True
        )
        # freeze normalization
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.x_mins, filter_spec, replace=False
        )
        filter_spec = eqx.tree_at(
            lambda tree: tree.physics.x_maxs, filter_spec, replace=False
        )
        return filter_spec

    # Move some of below to actually network implementation
    def freeze_physics_filter(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        for n in range(len(self.fields.layers)):
            filter_spec = eqx.tree_at(
                lambda tree: (
                    tree.fields.layers[n].weight, tree.fields.layers[n].bias
                ),
                filter_spec,
                replace=(True, True),
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
        return filter_spec
