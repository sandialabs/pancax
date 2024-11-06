from .base_element import BaseElement, ShapeFunctions
import jax
import jax.numpy as jnp


class Tet4Element(BaseElement):
    elementType = "tet4"
    degree = 1
    coordinates = jnp.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    vertexNodes = jnp.arange(4)
    faceNodes = jnp.array([
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
        [0, 2, 1]
    ])
    interiorNodes = None

    def __init__(self):
        pass

    def compute_shapes(self, nodalPoints, evaluationPoints):
        def shape_function_values(xi):
            N = jnp.array([
                1. - xi[0] - xi[1] - xi[2],
                xi[0],
                xi[1],
                xi[2]
            ])
            return N
        
        def shape_function_gradients(xi):
            grad_N_xi = jnp.array([
                [-1., -1., -1.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ])
            return grad_N_xi

        Ns = jax.vmap(shape_function_values, in_axes=(0,))(evaluationPoints)
        dNs = jax.vmap(shape_function_gradients, in_axes=(0,))(evaluationPoints)
        return ShapeFunctions(Ns, dNs)
