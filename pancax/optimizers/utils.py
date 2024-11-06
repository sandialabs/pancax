from typing import Optional
import equinox as eqx
import jax


def trainable_filter(
  params: any,
  # freeze_fields: Optional[bool] = False,
  freeze_properties: Optional[bool] = True,
  freeze_basis: Optional[bool] = False,
  # freeze_linear_layer: Optional[bool] = False
):
  # TODO there's some logic to work out here
  filter_spec = jax.tree_util.tree_map(
    lambda _: True, params
  )

  # filter_spec = eqx.tree_at(
  #   lambda tree: tree.fields,
  #   filter_spec,
  #   replace=not freeze_fields
  # )
  filter_spec = eqx.tree_at(
    lambda tree: tree.properties.prop_params,
    filter_spec,
    replace=not freeze_properties
  )

  if freeze_basis:
    try:
      filter_spec = eqx.tree_at(
        lambda tree: (tree.fields.basis.weight, tree.fields.basis.bias),
        filter_spec,
        replace=(not freeze_basis, not freeze_basis)
      )
      filter_spec = eqx.tree_at(
        lambda tree: (tree.fields.linear.weight, tree.fields.linear.bias),
        filter_spec,
        replace=(True, True)
      )
    except:
      raise AttributeError('This network does not have a basis!')
  
  return filter_spec