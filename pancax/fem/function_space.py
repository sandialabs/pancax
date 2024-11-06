from jaxtyping import Array, Float, Int
from .mesh import Mesh
from pancax.fem.quadrature_rules import QuadratureRule
from pancax.fem.elements.base_element import ShapeFunctions
from pancax.timer import Timer
import equinox as eqx
import jax
import jax.numpy as np


# TODO need to do checks on inputs to make sure they're compatable
class NonAllocatedFunctionSpace(eqx.Module):
    quadrature_rule: QuadratureRule
    shape_functions: ShapeFunctions

    def __init__(self, mesh: Mesh, q_rule: QuadratureRule) -> None:
        with Timer("NonAllocatedFunctionSpace.__init__"):
            self.quadrature_rule = q_rule
            self.shape_functions = mesh.parentElement.compute_shapes(
                mesh.parentElement.coordinates, q_rule.xigauss
            )

    def compute_field_gradient(self, u, X):
        """
        Takes in element level coordinates X and field u
        """
        grad_Ns = self.shape_function_gradients(X)
        return jax.vmap(lambda u, grad_N: u.T @ grad_N, in_axes=(None, 0))(u, grad_Ns)

    def evaluate_on_element(self, U, X, state, dt, props, func):
        """
        Takes in element level field, coordinates, states, etc.
        and evaluates the function func
        """
        Ns = self.shape_function_values(X)
        grad_Ns = self.shape_function_gradients(X)

        u_qs = jax.vmap(lambda u, N: u.T @ N, in_axes=(None, 0))(U, Ns)
        grad_u_qs = jax.vmap(lambda u, grad_N: u.T @ grad_N, in_axes=(None, 0))(U, grad_Ns)
        X_qs = jax.vmap(lambda X, N: X.T @ N, in_axes=(None, 0))(X, Ns)
        func_vals = jax.vmap(func, in_axes=(0, 0, 0, None, 0, None))(
            u_qs, grad_u_qs, state, props, X_qs, dt
        )
        return func_vals

    def integrate_on_element(self, U, X, state, dt, props, func):
        func_vals = self.evaluate_on_element(U, X, state, dt, props, func)
        JxWs = self.JxWs(X)
        return np.dot(JxWs, func_vals)

    def integrate_on_elements(self, U, X, state, dt, props, func):
        el_vals = jax.vmap(self.integrate_on_element, in_axes=(0, 0, 0, None, None, None))(
            U, X, state, dt, props, func
        )
        return np.sum(el_vals)

    def JxWs(self, X: Float[Array, "nn nd"]) -> Float[Array, "nq"]:
        Js = jax.vmap(lambda x, dN: (x.T @ dN).T, in_axes=(None, 0))(X, self.shape_functions.gradients)
        return jax.vmap(lambda J, w: np.linalg.det(J) * w, in_axes=(0, 0))(Js, self.quadrature_rule.wgauss)

    def shape_function_values(self, X: Float[Array, "nn nd"]) -> Float[Array, "nq nnpe"]:
        return self.shape_functions.values
    
    def shape_function_gradients(self, X: Float[Array, "nn nd"]) -> Float[Array, "nq nnpe nd"]:
        Js = jax.vmap(lambda x, dN: (x.T @ dN).T, in_axes=(None, 0))(X, self.shape_functions.gradients)
        Jinvs = jax.vmap(lambda J: np.linalg.inv(J), in_axes=(0,))(Js)
        return jax.vmap(lambda Jinv, dN: (Jinv @ dN.T).T, in_axes=(0, 0))(Jinvs, self.shape_functions.gradients)


class FunctionSpace(eqx.Module):
    """
    Data needed for calculus on functions in the discrete function space.

    In describing the shape of the attributes, ``ne`` is the number of
    elements in the mesh, ``nqpe`` is the number of quadrature points per
    element, ``npe`` is the number of nodes per element, and ``nd`` is the
    spatial dimension of the domain.

    :param shapes: Shape function values on each element, shape (ne, nqpe, npe)
    :param vols: Volume attributed to each quadrature point. That is, the 
        quadrature weight (on the parameteric element domain) multiplied by
        the Jacobian determinant of the map from the parent element to the
        element in the domain. Shape (ne, nqpe).
    :param shapeGrads: Derivatives of the shape functions with respect to the
        spatial coordinates of the domain. Shape (ne, nqpe, npe, nd).
    :param mesh: The ``Mesh`` object of the domain.
    :param quadratureRule: The ``QuadratureRule`` on which to sample the shape 
        functions.
    :param isAxisymmetric: boolean indicating if the function space data are 
        axisymmetric.
    """
    shapes: Float[Array, "ne nqpe npe"]
    vols: Float[Array, "ne nqpe"]
    shapeGrads: Float[Array, "ne nqpe npe nd"]
    # mesh: any
    conns: Int[Array, "ne nnpe"]
    quadratureRule: QuadratureRule
    isAxisymmetric: bool


def construct_function_space(mesh, quadratureRule, mode2D='cartesian'):
    """Construct a discrete function space.

    Parameters
    ----------
    :param mesh: The mesh of the domain.
    :param quadratureRule: The quadrature rule to be used for integrating on the
    domain.
    :param mode2D: A string indicating how the 2D domain is interpreted for
    integration. Valid values are ``cartesian`` and ``axisymmetric``.
    Axisymetric mode will include the factor of 2*pi*r in the ``vols``
    attribute.

    Returns
    -------
    The ``FunctionSpace`` object.
    """
    with Timer('construct_function_space'):
        # shapeOnRef = interpolants.compute_shapes(mesh.parentElement, quadratureRule.xigauss)
        shapeOnRef = mesh.parentElement.compute_shapes(mesh.parentElement.coordinates, quadratureRule.xigauss)
        return construct_function_space_from_parent_element(mesh, shapeOnRef, quadratureRule, mode2D)


def construct_function_space_from_parent_element(mesh, shapeOnRef, quadratureRule, mode2D='cartesian'):
    """
    Construct a function space with precomputed shape function data on the parent element.

    This version of the function space constructor is Jax-transformable,
    and in particular can be jitted. The computation of the shape function
    values and derivatives on the parent element is not transformable in
    general. However, the mapping of the shape function data to the elements in
    the mesh is transformable. One can precompute the parent element shape
    functions once and for all, and then use this special factory function to
    construct the function space and avoid the non-transformable part of the
    operation. The primary use case is for shape sensitivities: the coordinates
    of the mesh change, and we want Jax to pick up the sensitivities of the
    shape function derivatives in space to the coordinate changes
    (which occurs through the mapping from the parent element to the spatial
    domain).

    Parameters
    ----------
    :param mesh: The mesh of the domain.
    :param shapeOnRef: A tuple of the shape function values and gradients on the
    parent element, evaluated at the quadrature points. The caller must
    take care to ensure the shape functions are evaluated at the same
    points as contained in the ``quadratureRule`` parameter.
    :param quadratureRule: The quadrature rule to be used for integrating on the
        domain.
    :param mode2D: A string indicating how the 2D domain is interpreted for
    integration. See the default factory function for details.

    Returns
    -------
    The ``FunctionSpace`` object.
    """

    shapes = jax.vmap(lambda elConns, elShape: elShape, (0, None))(mesh.conns, shapeOnRef.values)

    shapeGrads = jax.vmap(map_element_shape_grads, (None, 0, None, None))(
        mesh.coords, mesh.conns, mesh.parentElement, shapeOnRef.gradients
    )

    if mode2D == 'cartesian':
        el_vols = compute_element_volumes
        isAxisymmetric = False
    elif mode2D == 'axisymmetric':
        el_vols = compute_element_volumes_axisymmetric
        isAxisymmetric = True
    vols = jax.vmap(el_vols, (None, 0, None, None, None, None))(
        mesh.coords, mesh.conns, mesh.parentElement, shapeOnRef.values, shapeOnRef.gradients, quadratureRule.wgauss
    )

    # return FunctionSpace(shapes, vols, shapeGrads, mesh, quadratureRule, isAxisymmetric)
    return FunctionSpace(shapes, vols, shapeGrads, mesh.conns, quadratureRule, isAxisymmetric)


def map_element_shape_grads(coordField, nodeOrdinals, parentElement, shapeGradients):
    # coords here should be 3 x 2
    # shapegrads shoudl be 3 x 2
    # need J to be 2 x 2 but be careful about transpose
    # below from Cthonios
    # J    = (X_el * ∇N_ξ)'
    # J_inv = inv(J)
    # ∇N_X = (J_inv * ∇N_ξ')'
    Xn = coordField.take(nodeOrdinals,0)
    Js = jax.vmap(lambda x, dN: (x.T @ dN).T, in_axes=(None, 0))(Xn, shapeGradients)
    Jinvs = jax.vmap(lambda J: np.linalg.inv(J), in_axes=(0,))(Js)
    return jax.vmap(lambda Jinv, dN: (Jinv @ dN.T).T, in_axes=(0, 0))(Jinvs, shapeGradients)


def compute_element_volumes(coordField, nodeOrdinals, parentElement, shapes, shapeGradients, weights):
    Xn = coordField.take(nodeOrdinals,0)
    Js = jax.vmap(lambda x, dN: (x.T @ dN).T, in_axes=(None, 0))(Xn, shapeGradients)
    return jax.vmap(lambda J, w: np.linalg.det(J) * w, in_axes=(0, 0))(Js, weights)


# TODO make this work for arbitrary elements
# currently not used for any PINN stuff
def compute_element_volumes_axisymmetric(coordField, nodeOrdinals, parentElement, shapes, shapeGradients, weights):
    vols = compute_element_volumes(coordField, nodeOrdinals, parentElement, shapes, weights)
    Xn = coordField.take(nodeOrdinals,0)
    Rs = shapes@Xn[:,0]
    return 2*np.pi*Rs*vols


def default_modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
    return elemGrads


def compute_field_gradient(functionSpace, nodalField, nodalCoords, modify_element_gradient=default_modify_element_gradient):
    return jax.vmap(compute_element_field_gradient, (None,None,0,0,0,0,None))(nodalField, nodalCoords, functionSpace.shapes, functionSpace.shapeGrads, functionSpace.vols, functionSpace.conns, modify_element_gradient)


def interpolate_to_points(functionSpace, nodalField):
    # return jax.vmap(interpolate_to_element_points, (None, 0, 0))(nodalField, functionSpace.shapes, functionSpace.mesh.conns)
    return jax.vmap(interpolate_to_element_points, (None, 0, 0))(nodalField, functionSpace.shapes, functionSpace.conns)



def integrate_over_block(functionSpace, U, X, stateVars, props, dt, func, block,
                         *params, modify_element_gradient=default_modify_element_gradient):
    """
    Integrates a density function over a block of the mesh.

    :param functionSpace: Function space object to do the integration with.
    :param U: The vector of dofs for the primal field in the functional.
    :param X: Nodal coordinates
    :param stateVars: Internal state variable array.
    :param dt: Current time increment
    :param func: Lagrangian density function to integrate, Must have the signature
        ``func(u, dudx, q, x, *params) -> scalar``, where ``u`` is the primal field, ``q`` is the
        value of the internal variables, ``x`` is the current point coordinates, and ``*params`` is
        a variadic set of additional parameters, which correspond to the ``*params`` argument.
        block: Group of elements to integrate over. This is an array of element indices. For
        performance, the elements within the block should be numbered consecutively.
    :param modify_element_gradient: Optional function that modifies the gradient at the element level.
        This can be to set the particular 2D mode, and additionally to enforce volume averaging
        on the gradient operator. This is a keyword-only argument.

    Returns
    A scalar value for the integral of the density functional ``func`` integrated over the
    block of elements.
    """
    # below breaks the sphinx doc stuff
    # :param *params: Optional parameter fields to pass into Lagrangian density function. These are 
    
    
    vals = evaluate_on_block(functionSpace, U, X, stateVars, props, dt, func, block, *params, modify_element_gradient=modify_element_gradient)
    return np.dot(vals.ravel(), functionSpace.vols[block].ravel())


# def evaluate_on_block(functionSpace, U, stateVars, dt, props, func, block,
#                       *params, modify_element_gradient=default_modify_element_gradient):
def evaluate_on_block(functionSpace, U, X, stateVars, dt, props, func, block,
                      *params, modify_element_gradient=default_modify_element_gradient):
    """Evaluates a density function at every quadrature point in a block of the mesh.

    :param functionSpace: Function space object to do the evaluation with.
    :param U: The vector of dofs for the primal field in the functional.
    :param X: Nodal coordinates
    :param stateVars: Internal state variable array.
    :param dt: Current time increment
    :param func: Lagrangian density function to evaluate, Must have the signature
        ```func(u, dudx, q, x, *params) -> scalar```, where ```u``` is the primal field, ```q``` is the 
        value of the internal variables, ```x``` is the current point coordinates, and ```*params``` is 
        a variadic set of additional parameters, which correspond to the ```*params``` argument.
    :param block: Group of elements to evaluate over. This is an array of element indices. For 
        performance, the elements within the block should be numbered consecutively.
    
    :param modify_element_gradient: Optional function that modifies the gradient at the element level. 
        This can be to set the particular 2D mode, and additionally to enforce volume averaging 
        on the gradient operator. This is a keyword-only argument.

    Returns 
    An array of shape (numElements, numQuadPtsPerElement) that contains the scalar values of the 
    density functional ```func``` at every quadrature point in the block.
    """
    # below breaks sphinx doc stuff
    # :param *params: Optional parameter fields to pass into Lagrangian density function. These are 
    #     represented as a single value per element. 
    fs = functionSpace
    compute_elem_values = jax.vmap(evaluate_on_element, (None, None, 0, None, None, 0, 0, 0, 0, None, None, *tuple(0 for p in params)))
    
    blockValues = compute_elem_values(U, X, stateVars[block], props, dt, fs.shapes[block],
                                      fs.shapeGrads[block], fs.vols[block],
                                      fs.conns[block], func, modify_element_gradient, *params)
    return blockValues


def integrate_element_from_local_field(elemNodalField, elemNodalCoords, elemStates, dt, elemShapes, elemShapeGrads, elemVols, func, modify_element_gradient=default_modify_element_gradient):
    """
    Integrate over element with element nodal field as input.
    This allows element residuals and element stiffness matrices to computed.
    """
    elemVals = jax.vmap(interpolate_to_point, (None,0))(elemNodalField, elemShapes)
    elemGrads = jax.vmap(compute_quadrature_point_field_gradient, (None,0))(elemNodalField, elemShapeGrads)
    elemGrads = modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalField, elemNodalCoords)
    elemPoints = jax.vmap(interpolate_to_point, (None,0))(elemNodalCoords, elemShapes)
    fVals = jax.vmap(func, (0, 0, 0, 0, None))(elemVals, elemGrads, elemStates, elemPoints, dt)
    return np.dot(fVals, elemVols)


def compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConnectivity, modify_element_gradient):
    elemNodalDisps = U[elemConnectivity]
    elemGrads = jax.vmap(compute_quadrature_point_field_gradient, (None, 0))(elemNodalDisps, elemShapeGrads)
    elemNodalCoords = coords[elemConnectivity]
    elemGrads = modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords)
    return elemGrads


def compute_quadrature_point_field_gradient(u, shapeGrad):
    dg = np.tensordot(u, shapeGrad, axes=[0,0])
    return dg


def interpolate_to_point(elementNodalValues, shape):
    return np.dot(shape, elementNodalValues)


def interpolate_to_element_points(U, elemShapes, elemConnectivity):
    elemU = U[elemConnectivity]
    return jax.vmap(interpolate_to_point, (None, 0))(elemU, elemShapes)


def integrate_element(U, coords, elemStates, elemShapes, elemShapeGrads, elemVols, elemConn, func, modify_element_gradient):
    elemVals = interpolate_to_element_points(U, elemShapes, elemConn)
    elemGrads = compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConn, modify_element_gradient)
    elemXs = interpolate_to_element_points(coords, elemShapes, elemConn)
    fVals = jax.vmap(func)(elemVals, elemGrads, elemStates, elemXs)
    return np.dot(fVals, elemVols)


def evaluate_on_element(U, coords, elemStates, props, dt, elemShapes, elemShapeGrads, elemVols, elemConn, kernelFunc, modify_element_gradient, *params):
    elemVals = interpolate_to_element_points(U, elemShapes, elemConn)
    elemGrads = compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConn, modify_element_gradient)
    elemXs = interpolate_to_element_points(coords, elemShapes, elemConn)
    vmapArgs = 0, 0, 0, None, 0, None, *tuple(None for p in params)
    fVals = jax.vmap(kernelFunc, vmapArgs)(elemVals, elemGrads, elemStates, props, elemXs, dt, *params)
    return fVals


def project_quadrature_field_to_element_field(functionSpace, quadField):
    return jax.vmap(average_quadrature_field_over_element)(quadField, functionSpace.vols)


def average_quadrature_field_over_element(elemQPData, vols):
    S = np.tensordot(vols, elemQPData, axes=[0,0])
    elVol = np.sum(vols)
    return S/elVol


def get_nodal_values_on_edge(functionSpace, nodalField, edge):
    """
    Get nodal values of a field on an element edge.

    :param functionSpace: a FunctionSpace object
    :param nodalField: The nodal vector defined over the mesh (shape is number of
        nodes by number of field components)
    :param edge: tuple containing the element number containing the edge and the
        permutation (0, 1, or 2) of the edge within the triangle
    """
    edgeNodes = functionSpace.mesh.parentElement.faceNodes[edge[1], :]
    nodes = functionSpace.mesh.conns[edge[0], edgeNodes]
    return nodalField[nodes]


def interpolate_nodal_field_on_edge(functionSpace, U, interpolationPoints, edge):
    """
    Interpolate a nodal field to specified points on an element edge.

    :param functionSpace: a FunctionSpace object
    :param U: the nodal values array
    :param interpolationPoints: coordinates of points (in the 1D parametric space) to
        interpolate to
    :param edge: tuple containing the element number containing the edge and the
        permutation (0, 1, or 2) of the edge within the triangle
    """
    # edgeShapes = interpolants.compute_shapes(functionSpace.mesh.parentElement1d, interpolationPoints)
    edgeShapes = functionSpace.mesh.parentElement1d.compute_shapes(
        functionSpace.mesh.parentElement1d.coordinates, interpolationPoints
    )
    edgeU = get_nodal_values_on_edge(functionSpace, U, edge)
    return edgeShapes.values.T@edgeU


def integrate_function_on_edge(functionSpace, func, U, quadRule, edge):
    uq = interpolate_nodal_field_on_edge(functionSpace, U, quadRule.xigauss, edge)
    Xq = interpolate_nodal_field_on_edge(functionSpace, functionSpace.mesh.coords, quadRule.xigauss, edge)
    edgeCoords = Mesh.get_edge_coords(functionSpace.mesh, edge)
    _, normal, jac = Mesh.compute_edge_vectors(functionSpace.mesh, edgeCoords)
    integrand = jax.vmap(func, (0, 0, None))(uq, Xq, normal)
    return np.dot(integrand, jac*quadRule.wgauss)


def integrate_function_on_edges(functionSpace, func, U, quadRule, edges):
    integrate_on_edges = jax.vmap(integrate_function_on_edge, (None, None, None, None, 0))
    return np.sum(integrate_on_edges(functionSpace, func, U, quadRule, edges))
