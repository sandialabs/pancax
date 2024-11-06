from .elm import ELM, ELM2
from .field_property_pair import FieldPropertyPair
from .mlp import Linear
from .mlp import MLP
from .mlp import MLPBasis
from .properties import *
from .rbf import RBFBasis
from .rbf import rbf_normalization
from .initialization import *

def Network(network_type, *args, **kwargs):
  return network_type(*args, **kwargs)

