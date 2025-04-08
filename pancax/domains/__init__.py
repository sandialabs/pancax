from .base import BaseDomain
from .collocation_domain import CollocationDomain
from .delta_pinn_domain import DeltaPINNDomain
from .variational_domain import VariationalDomain

__all__ = [
    "BaseDomain",
    "CollocationDomain",
    "DeltaPINNDomain",
    "VariationalDomain"
]
