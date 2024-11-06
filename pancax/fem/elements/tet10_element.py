from .base_element import BaseElement, ShapeFunctions
import jax
import jax.numpy as jnp


class Tet10Element(BaseElement):
    elementType = "tet10"
    degree = 2
    coordinates = jnp.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.5, 0., 0.],
        [0.5, 0.5, 0.],
        [0., 0.5, 0.],
        [0., 0., 0.5],
        [0.5, 0., 0.5],
        [0., 0.5, 0.5]
    ])
    vertexNodes = jnp.arange(10)
    faceNodes = jnp.array([
        [0, 4, 1, 8, 3, 7],
        [1, 5, 2, 9, 3, 7],
        [0, 7, 3, 9, 2, 6],
        [0, 6, 2, 5, 1, 4]
    ])
    interiorNodes = None

    def __init__(self):
        pass

    def compute_shapes(self, nodalPoints, evaluationPoints):
        def shape_function_values(xi):
            t0 = 1 - xi[0] - xi[1] - xi[2]
            t1 = xi[0]
            t2 = xi[1]
            t3 = xi[2]
            N = jnp.array([
                t0 * (2 * t0 - 1),
                t1 * (2 * t1 - 1),
                t2 * (2 * t2 - 1),
                t3 * (2 * t3 - 1),
                4 * t0 * t1,
                4 * t1 * t2,
                4 * t2 * t0,
                4 * t0 * t3,
                4 * t1 * t3,
                4 * t2 * t3
            ])
            return N
        
        def shape_function_gradients(xi):
            t0 = 1 - xi[0] - xi[1] - xi[2]
            t1 = xi[0]
            t2 = xi[1]
            t3 = xi[2]
            grad_N_xi = jnp.array([
                [1 - 4 * t0, 1 - 4 * t0, 1 - 4 * t0],
                [4 * t1 - 1, 0, 0],
                [0, 4 * t2 - 1, 0],
                [0, 0, 4 * t3 - 1],
                [4 * (t0-t1), -4 * t1, -4 * t1],
                [4 * t2, 4 * t1, 0],
                [-4 * t2, 4 * (t0 - t2), -4 * t2],
                [-4 * t3, -4 * t3, 4 * (t0 - t3)],
                [4 * t3, 0, 4 * t1],
                [0, 4 * t3, 4 * t2]
            ])
            return grad_N_xi

        Ns = jax.vmap(shape_function_values, in_axes=(0,))(evaluationPoints)
        dNs = jax.vmap(shape_function_gradients, in_axes=(0,))(evaluationPoints)
        return ShapeFunctions(Ns, dNs)
