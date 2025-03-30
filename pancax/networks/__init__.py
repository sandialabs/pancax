from .elm import ELM, ELM2
from .fields import Field
from .field_physics_pair import FieldPhysicsPair
from .ml_dirichlet_field import MLDirichletField
from .mlp import Linear
from .mlp import MLP
from .mlp import MLPBasis
from .parameters import Parameters
from .rbf import RBFBasis
from .rbf import rbf_normalization
from .initialization import *

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
  "Parameters"
]
