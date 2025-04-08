from jaxtyping import Array, Float
from typing import Callable, List, Optional
import equinox as eqx


BCFunc = Callable[[Float[Array, "nd"], float], float]


class DirichletBC(eqx.Module):
    """
    :param nodeSet: A name for a nodeset in the mesh
    :param component: The dof to apply the dirichlet bc to
    :param function: A function f(x, t) = u that gives the value
        to enforce on the (nodeset, component) of a field.
        This defaults to the zero function
    """

    nodeSet: str
    component: int
    function: Optional[BCFunc] = lambda x, t: 0.0

    def coordinates(self, mesh):
        nodes = mesh.nodeSets[self.nodeSet]
        coords = mesh.coords
        return coords[nodes, :]


class DirichletBCSet(eqx.Module):
    bcs: List[DirichletBC]
