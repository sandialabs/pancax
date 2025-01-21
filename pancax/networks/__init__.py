from .elm import ELM, ELM2
from .field_physics_pair import FieldPhysicsPair
from .mlp import Linear
from .mlp import MLP
from .mlp import MLPBasis
from .properties import *
from .rbf import RBFBasis
from .rbf import rbf_normalization
from .initialization import *

def Network(network_type, *args, **kwargs):
  return network_type(*args, **kwargs)

