from .base_element import BaseElement, ShapeFunctions
from .base_element import get_lobatto_nodes_1d, vander1d
from jaxtyping import Array, Float
import jax.numpy as jnp
import numpy as np


class LineElement(BaseElement):
    elementType = "line"
    degree: int
    coordinates: Float[Array, "nn nd"]
    vertexNodes: any
    faceNodes = None
    interiorNodes: any

    def __init__(self, degree):
        self.degree = degree
        self.coordinates = get_lobatto_nodes_1d(degree)
        self.vertexNodes = jnp.array([0, degree], dtype=jnp.int32)
        self.interiorNodes = jnp.arange(1, degree, dtype=jnp.int32)

    def compute_shapes(self, nodalPoints, evaluationPoints):
        A,_ = vander1d(nodalPoints, self.degree)
        nf, nfx = vander1d(evaluationPoints, self.degree)
        shape = np.linalg.solve(A.T, nf.T)
        dshape = np.linalg.solve(A.T, nfx.T)
        return ShapeFunctions(jnp.array(shape), jnp.array(dshape))
