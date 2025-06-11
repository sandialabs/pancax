from .fields import Field
from .field_physics_pair import FieldPhysicsPair
from .initialization import trunc_init
from .ml_dirichlet_field import MLDirichletField
from .mlp import Linear
from .mlp import MLP
from .mlp import MLPBasis
from .parameters import Parameters
from .resnet import ResNet


def Network(network_type, *args, **kwargs):
    return network_type(*args, **kwargs)


__all__ = [
    "Field",
    "FieldPhysicsPair",
    "Linear",
    "MLDirichletField",
    "MLP",
    "MLPBasis",
    "Network",
    "Parameters",
    "ResNet",
    "trunc_init"
]
