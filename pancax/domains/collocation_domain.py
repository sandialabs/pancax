from .base import BaseDomain
from jaxtyping import Array, Float
from pancax.fem import Mesh
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp
import numpy as np


class CollocationDomain(BaseDomain):
    mesh_file: str
    mesh: Mesh
    coords: Float[Array, "nn nd"]
    times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]

    def __init__(
        self, mesh_file: str, times: Float[Array, "nt"],
        p_order: Optional[int] = 1
    ) -> None:
        super().__init__(mesh_file, times, p_order=p_order)


class CollocationDataLoader(eqx.Module):
    indices: np.ndarray
    inputs: Float[Array, "bs ni"]
    outputs: Float[Array, "bs no"]

    def __init__(
        self,
        domain: CollocationDomain,
        num_fields: int
    ) -> None:
        inputs = []

        # For now, just a simple collection of mesh coordinates
        # TODO add sampling strategies
        coords = domain.coords
        ones = jnp.ones((coords.shape[0], 1))
        for time in domain.times:
            times = time * ones
            temp = jnp.hstack((coords, times))
            inputs.append(temp)

        inputs = jnp.vstack(inputs)
        outputs = jnp.zeros((inputs.shape[0], num_fields))

        indices = np.arange(len(inputs))

        self.indices = indices
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def dataloader(self, batch_size: int):
        perm = np.random.permutation(self.indices)
        start = 0
        end = batch_size
        while end <= len(self):
            batch_perm = perm[start:end]
            yield self.inputs[batch_perm], self.outputs[batch_perm]
            start = end
            end = start + batch_size
