from .base_loss_function import BaseLossFunction
from jax import vmap
from typing import Optional
import jax.numpy as jnp


class FullFieldDataLoss(BaseLossFunction):
    weight: float

    def __init__(self, weight: Optional[float] = 1.0):
        self.weight = weight

    def __call__(self, params, problem, inputs, outputs):
        field_network, _, _ = params
        n_dims = problem.coords.shape[1]
        xs = inputs[:, 0:n_dims]
        ts = inputs[:, n_dims]
        u_pred = vmap(problem.physics.field_values, in_axes=(None, 0, 0))(
            field_network, xs, ts
        )
        # TODO add output normalization
        loss = jnp.square(u_pred - outputs).mean()
        aux = {"field_data_loss": loss}
        return self.weight * loss, aux
