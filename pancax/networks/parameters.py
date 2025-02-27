from .base import BasePancaxModel
from jaxtyping import Array, Float
from typing import Optional, Union
import equinox as eqx
import jax.tree_util as jtu


State = Union[Float[Array, "nt ne nq ns"], eqx.Module, None]


class Parameters(BasePancaxModel):
  """
  Data structure for storing all parameters
  needed for a model

  :param field: field network parameters object
  :param physics: physics object
  :param state: state object
  """
  field: eqx.Module
  physics: eqx.Module
  state: State

  def __init__(self, field, physics, state: Optional[State] = None):
    self.field = field
    self.physics = physics
    self.state = state
  
  def __iter__(self):
    return iter((self.field, self.physics))

  def freeze_fields_filter(self):
    filter_spec = jtu.tree_map(lambda _: False, self)
    filter_spec = eqx.tree_at(
      lambda tree: tree.physics,
      filter_spec,
      replace=True
    )
    # freeze normalization
    filter_spec = eqx.tree_at(
      lambda tree: tree.physics.x_mins,
      filter_spec,
      replace=False
    )
    filter_spec = eqx.tree_at(
      lambda tree: tree.physics.x_maxs,
      filter_spec,
      replace=False
    )
    return filter_spec

  # Move some of below to actually network implementation
  def freeze_physics_filter(self):
    filter_spec = jtu.tree_map(lambda _: False, self)
    for n in range(len(self.fields.layers)):
      filter_spec = eqx.tree_at(
        lambda tree: (tree.fields.layers[n].weight, tree.fields.layers[n].bias),
        filter_spec,
        replace=(True, True),
      )
    return filter_spec

  def freeze_physics_normalization_filter(self):
    filter_spec = jtu.tree_map(lambda _: True, self)
    filter_spec = eqx.tree_at(
      lambda tree: tree.physics.x_mins,
      filter_spec,
      replace=False
    )
    filter_spec = eqx.tree_at(
      lambda tree: tree.physics.x_maxs,
      filter_spec,
      replace=False
    )
    return filter_spec
