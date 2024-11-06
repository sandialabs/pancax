from .base_element import BaseElement, ShapeFunctions
import jax
import jax.numpy as jnp


class Hex8Element(BaseElement):
    elementType = "hex8"
    degree = 1
    coordinates = jnp.array(
        [[-1.0, -1.0, -1.0],
         [ 1.0, -1.0, -1.0],
         [ 1.0,  1.0, -1.0],
         [-1.0,  1.0, -1.0],
         [-1.0, -1.0, 1.0],
         [ 1.0, -1.0, 1.0],
         [ 1.0,  1.0, 1.0],
         [-1.0,  1.0, 1.0]]
    )
    vertexNodes = jnp.arange(8)
    faceNodes = jnp.array(
        [[0, 1, 5, 6],
         [1, 2, 6, 5],
         [2, 3, 7, 6],
         [0, 4, 7, 3],
         [0, 3, 2, 1],
         [4, 5, 6, 7]],
        dtype=jnp.int32
    )
    interiorNodes = None

    def __init__(self):
        pass

    # TODO what to do about NodalPoints?
    def compute_shapes(self, nodalPoints, evaluationPoints):
        def shape_function_values(xi):
            N = (1. / 8.) * jnp.array([
                (1. - xi[0]) * (1. - xi[1]) * (1. - xi[2]),
                (1. + xi[0]) * (1. - xi[1]) * (1. - xi[2]),
                (1. + xi[0]) * (1. + xi[1]) * (1. - xi[2]),
                (1. - xi[0]) * (1. + xi[1]) * (1. - xi[2]),
                (1. - xi[0]) * (1. - xi[1]) * (1. + xi[2]),
                (1. + xi[0]) * (1. - xi[1]) * (1. + xi[2]),
                (1. + xi[0]) * (1. + xi[1]) * (1. + xi[2]),
                (1. - xi[0]) * (1. + xi[1]) * (1. + xi[2])]
            )
            return N
        
        def shape_function_gradients(xi):
            grad_N_xi = (1. / 8.) * jnp.array(
                [
                    [
                        -(1. - xi[1]) * (1. - xi[2]), 
                        -(1. - xi[0]) * (1. - xi[2]), 
                        -(1. - xi[0]) * (1. - xi[1])
                    ],
                    [
                         (1. - xi[1]) * (1. - xi[2]), 
                        -(1. + xi[0]) * (1. - xi[2]),
                        -(1. + xi[0]) * (1. - xi[1])
                    ],
                    [
                         (1. + xi[1]) * (1. - xi[2]), 
                         (1. + xi[0]) * (1. - xi[2]),
                        -(1. + xi[0]) * (1. + xi[1]) 
                    ],
                    [
                        -(1. + xi[1]) * (1. - xi[2]), 
                         (1. - xi[0]) * (1. - xi[2]),
                        -(1. - xi[0]) * (1. + xi[1])
                    ],
                    # positive z now
                    [
                        -(1. - xi[1]) * (1. + xi[2]), 
                        -(1. - xi[0]) * (1. + xi[2]), 
                         (1. - xi[0]) * (1. - xi[1])
                    ],
                    [
                         (1. - xi[1]) * (1. + xi[2]), 
                        -(1. + xi[0]) * (1. + xi[2]),
                         (1. + xi[0]) * (1. - xi[1])
                    ],
                    [
                         (1. + xi[1]) * (1. + xi[2]), 
                         (1. + xi[0]) * (1. + xi[2]),
                         (1. + xi[0]) * (1. + xi[1]) 
                    ],
                    [
                        -(1. + xi[1]) * (1. + xi[2]), 
                         (1. - xi[0]) * (1. + xi[2]),
                         (1. - xi[0]) * (1. + xi[1])
                    ]
                 ]
            )
            return grad_N_xi

        Ns = jax.vmap(shape_function_values, in_axes=(0,))(evaluationPoints)
        dNs = jax.vmap(shape_function_gradients, in_axes=(0,))(evaluationPoints)
        return ShapeFunctions(Ns, dNs)
