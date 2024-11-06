from .base_element import BaseElement, ShapeFunctions
import jax
import jax.numpy as jnp


class Quad4Element(BaseElement):
    elementType = "quad4"
    degree = 1
    coordinates = jnp.array(
        [[-1.0, -1.0],
         [ 1.0, -1.0],
         [ 1.0,  1.0],
         [-1.0,  1.0]]
    )
    vertexNodes = jnp.arange(3)
    faceNodes = jnp.array(
        [[0, 1],
         [1, 2],
         [2, 3],
         [3, 0]],
        dtype=jnp.int32
    )
    interiorNodes = None

    def __init__(self):
        pass

    # TODO what to do about NodalPoints?
    def compute_shapes(self, nodalPoints, evaluationPoints):
        def shape_function_values(xi):
            N = (1. / 4.) * jnp.array([
                (1. - xi[0]) * (1. - xi[1]),
                (1. + xi[0]) * (1. - xi[1]),
                (1. + xi[0]) * (1. + xi[1]),
                (1. - xi[0]) * (1. + xi[1])]
            )
            return N
        
        def shape_function_gradients(xi):
            grad_N_xi = (1. / 4.) * jnp.array(
                [[-(1. - xi[1]), -(1. - xi[0])],
                 [(1. - xi[1]), -(1. + xi[0])],
                 [(1. + xi[1]), (1. + xi[0])],
                 [-(1. + xi[1]), (1. - xi[0])]]
            )
            return grad_N_xi

        Ns = jax.vmap(shape_function_values, in_axes=(0,))(evaluationPoints)
        dNs = jax.vmap(shape_function_gradients, in_axes=(0,))(evaluationPoints)
        return ShapeFunctions(Ns, dNs)
