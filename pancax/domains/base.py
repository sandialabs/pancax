from jaxtyping import Array, Float
from pancax.fem import Mesh, SimplexTriElement
from pancax.fem import create_higher_order_mesh_from_simplex_mesh
from pancax.fem import read_exodus_mesh
from pancax.timer import Timer
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp


class SimulationTimesNotUniqueException(Exception):
    pass


class SimulationTimesNotStrictlyIncreasingException(Exception):
    pass


class BaseDomain(eqx.Module):
    mesh_file: str
    mesh: Mesh
    coords: Float[Array, "nn nd"]
    times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]

    def __init__(
        self, mesh_file: str, times: Float[Array, "nt"],
        p_order: Optional[int] = 1
    ) -> None:
        with Timer("Reading Mesh..."):
            mesh = read_exodus_mesh(mesh_file)
            # if tri mesh, we can make it higher order from lower order
            if type(mesh.parentElement) is SimplexTriElement:
                mesh = create_higher_order_mesh_from_simplex_mesh(
                    mesh, p_order, copyNodeSets=True
                )
            else:
                print(
                    "WARNING: Ignoring polynomial \
                    order flag for non tri mesh"
                )

            # checking provided simulation times are unique
            if len(times) != len(set(times.tolist())):
                raise SimulationTimesNotUniqueException()

            # checking provided times are strictly increasing
            for i in range(len(times) - 1):
                if times[i] >= times[i + 1]:
                    raise SimulationTimesNotStrictlyIncreasingException()

            self.mesh_file = mesh_file
            self.mesh = mesh
            self.coords = jnp.array(mesh.coords)
            self.times = times
