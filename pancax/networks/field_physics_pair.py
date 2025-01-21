import equinox as eqx
import jax.tree_util as jtu


class BasePancaxModel(eqx.Module):
  """
  Base class for pancax model parameters.

  This includes a few helper methods
  """
  def serialise(self, base_name, epoch):
    file_name = f'{base_name}_{str(epoch).zfill(7)}.eqx'
    print(f'Serialising current parameters to {file_name}')
    eqx.tree_serialise_leaves(file_name, self)


class FieldPhysicsPair(BasePancaxModel):
  """
  Data structure for storing a set of field network
  parameters and a physics object

  :param fields: field network parameters object
  :param physics: physics object
  """
  fields: eqx.Module
  physics: eqx.Module

  def __iter__(self):
    """
    Iterator for user friendliness
    """
    return iter((self.fields, self.physics))

  def freeze_fields_filter(self):
    filter_spec = jtu.tree_map(lambda _: False, self)
    filter_spec = eqx.tree_at(
        # lambda tree: tree.properties.prop_params,
        lambda tree: tree.physics,
        filter_spec,
        replace=True
    )
    return filter_spec

  def freeze_physics_filter(self):
    filter_spec = jtu.tree_map(lambda _: False, self)
    for n in range(len(self.fields.layers)):
      filter_spec = eqx.tree_at(
        lambda tree: (tree.fields.layers[n].weight, tree.fields.layers[n].bias),
        filter_spec,
        replace=(True, True),
      )
    return filter_spec
