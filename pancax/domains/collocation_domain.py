from .base import BaseDomain
from jaxtyping import Array, Float
from pancax.fem import Mesh
from typing import Optional, Union


class CollocationDomain(BaseDomain):
  mesh_file: str
  mesh: Mesh
  coords: Float[Array, "nn nd"]
  times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]

  def __init__(
      self, 
      mesh_file: str, 
      times: Float[Array, "nt"],
      p_order: Optional[int] = 1
  ) -> None:
    super().__init__(mesh_file, times, p_order=p_order)
