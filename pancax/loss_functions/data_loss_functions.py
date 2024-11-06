from .base_loss_function import BaseLossFunction
from jax import vmap
from typing import Optional
import jax.numpy as jnp


class FullFieldDataLoss(BaseLossFunction):
  weight: float

  def __init__(
    self, 
    weight: Optional[float] = 1.0
  ):
    self.weight = weight

  def __call__(self, params, domain):
    field_network, _ = params
    n_dims = domain.coords.shape[1]
    xs = domain.field_data.inputs[:, 0:n_dims]
    # TODO need time normalization
    ts = domain.field_data.inputs[:, n_dims]
    # TODO below is currenlty the odd ball for the field_value API
    u_pred = vmap(domain.physics.field_values, in_axes=(None, 0, 0))(
      field_network, xs, ts
    )

    # TODO add output normalization
    loss = jnp.square(u_pred - domain.field_data.outputs).mean()
    aux = {'field_data_loss': loss}
    return self.weight * loss, aux
