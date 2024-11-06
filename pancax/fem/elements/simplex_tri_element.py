from .base_element import BaseElement, ShapeFunctions
from .base_element import get_lobatto_nodes_1d, vander2d
from jaxtyping import Array, Float
import jax.numpy as jnp
import numpy as np


class SimplexTriElement(BaseElement):
    elementType = "simplex tri"
    degree: int
    coordinates: Float[Array, "nn nd"]
    vertexNodes: any
    faceNodes: any
    interiorNodes: any

    def __init__(self, degree):
        self.degree = degree

        lobattoPoints = get_lobatto_nodes_1d(degree)
        nPoints = int((degree + 1)*(degree + 2)/2)
        points = np.zeros((nPoints, 2))
        point = 0
        for i in range(degree):
            for j in range(degree + 1 - i):
                k = degree - i - j
                points[point, 0] = (1.0 + 2.0*lobattoPoints[k] - lobattoPoints[j] - lobattoPoints[i])/3.0
                points[point, 1] = (1.0 + 2.0*lobattoPoints[j] - lobattoPoints[i] - lobattoPoints[k])/3.0
                point += 1
        self.coordinates = jnp.asarray(points)

        self.vertexNodes = jnp.array([0, degree, nPoints - 1], dtype=jnp.int32)

        ii = np.arange(degree + 1)
        jj = np.cumsum(np.flip(ii)) + ii
        kk = np.flip(jj) - ii
        self.faceNodes = jnp.array((ii,jj,kk), dtype=jnp.int32)

        interiorNodes = [i for i in range(nPoints) if i not in self.faceNodes.ravel()]
        self.interiorNodes = np.array(interiorNodes, dtype=np.int32)

    def compute_shapes(self, nodalPoints, evaluationPoints):
        A, _, _ = vander2d(nodalPoints, self.degree)
        nf, nfx, nfy = vander2d(evaluationPoints, self.degree)
        shapes = np.linalg.solve(A.T, nf.T).T
        dshapes = np.zeros(shapes.shape + (2,)) # shape is (nQuadPoints, nNodes, 2)
        dshapes[:, :, 0] = np.linalg.solve(A.T, nfx.T).T
        dshapes[:, :, 1] = np.linalg.solve(A.T, nfy.T).T
        return ShapeFunctions(jnp.asarray(shapes), jnp.asarray(dshapes))
