from .base import BasePancaxModel
import equinox as eqx
import jax.tree_util as jtu


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
