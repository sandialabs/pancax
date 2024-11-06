from .base_element import BaseElement, ShapeFunctions
import jax
import jax.numpy as jnp


class Quad9Element(BaseElement):
    elementType = "quad9"
    degree = 2
    coordinates = jnp.array(
        [[-1.0, -1.0],
         [ 1.0, -1.0],
         [ 1.0,  1.0],
         [-1.0,  1.0],
         [ 0.0, -1.0],
         [ 1.0,  0.0],
         [ 0.0,  1.0],
         [-1.0,  0.0],
         [ 0.0,  0.0]]

    )
    vertexNodes = jnp.arange(8)
    faceNodes = jnp.array(
        [[0, 4, 1],
         [1, 5, 2],
         [2, 6, 3],
         [3, 7, 0]],
        dtype=jnp.int32
    )
    interiorNodes = jnp.array([8], dtype=jnp.int32)

    def __init__(self):
        pass

    # TODO what to do about NodalPoints?
    def compute_shapes(self, nodalPoints, evaluationPoints):
        def shape_function_values(xi_points):
            xi, eta = xi_points[0], xi_points[1]
            xi_sq, eta_sq = jnp.power(xi_points[0], 2), jnp.power(xi_points[1], 2)

            N = jnp.array([
                0.25 * (xi_sq - xi) * (eta_sq - eta),
                0.25 * (xi_sq + xi) * (eta_sq - eta),
                0.25 * (xi_sq + xi) * (eta_sq + eta),
                0.25 * (xi_sq - xi) * (eta_sq + eta),
                0.5 * (eta_sq - eta) * (1. - xi_sq),
                0.5 * (xi_sq + xi) * (1. - eta_sq),
                0.5 * (eta_sq + eta) * (1. - xi_sq),
                0.5 * (xi_sq - xi) * (1. - eta_sq),
                (1. - xi_sq) * (1. - eta_sq)
            ])
            return N

        
        def shape_function_gradients(xi_points):
            xi, eta = xi_points[0], xi_points[1]
            xi_sq, eta_sq = jnp.power(xi_points[0], 2), jnp.power(xi_points[1], 2)
            grad_N_xi = jnp.array([[0.25 * (2. * xi - 1.) * (eta_sq - eta), 0.25 * (xi_sq - xi) * (2. * eta - 1.)],
                                [0.25 * (2. * xi + 1.) * (eta_sq - eta), 0.25 * (xi_sq + xi) * (2. * eta - 1.)],
                                [0.25 * (2. * xi + 1.) * (eta_sq + eta), 0.25 * (xi_sq + xi) * (2. * eta + 1.)],
                                [0.25 * (2. * xi - 1.) * (eta_sq + eta), 0.25 * (xi_sq - xi) * (2. * eta + 1.)],
                                [0.5 * (eta_sq - eta) * (-2. * xi), 0.5 * (2. * eta - 1.) * (1. - xi_sq)],
                                [0.5 * (2. * xi + 1.) * (1. - eta_sq), 0.5 * (xi_sq + xi) * (-2. * eta)],
                                [0.5 * (eta_sq + eta) * (-2. * xi), 0.5 * (2. * eta + 1.) * (1. - xi_sq)],
                                [0.5 * (2. * xi - 1.) * (1. - eta_sq), 0.5 * (xi_sq - xi) * (-2. * eta)],
                                [-2. * xi * (1. - eta_sq), -2. * eta * (1. - xi_sq)]])
            return grad_N_xi


        Ns = jax.vmap(shape_function_values, in_axes=(0,))(evaluationPoints)
        dNs = jax.vmap(shape_function_gradients, in_axes=(0,))(evaluationPoints)
        return ShapeFunctions(Ns, dNs)
