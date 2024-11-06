from abc import abstractmethod
from jaxtyping import Array, Float, Int
from scipy import special
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def get_lobatto_nodes_1d(degree):
    p = np.polynomial.Legendre.basis(degree, domain=[0.0, 1.0])
    dp = p.deriv()
    xInterior = dp.roots()
    xn = jnp.hstack((jnp.array([0.0]), xInterior, jnp.array([1.0])))
    return xn


def pascal_triangle_monomials(degree):
    p = []
    q = []
    for i in range(1, degree + 2):
        monomialIndices = list(range(i))
        p += monomialIndices
        monomialIndices.reverse()
        q += monomialIndices
    return np.column_stack((q,p))


def vander1d(x, degree):
    x = np.asarray(x)
    A = np.zeros((x.shape[0], degree + 1))
    dA = np.zeros((x.shape[0], degree + 1))
    domain = [0.0, 1.0]
    for i in range(degree + 1):
        p = np.polynomial.Legendre.basis(i, domain=domain) 
        p *= np.sqrt(2.0*i + 1.0) # keep polynomial orthonormal
        A[:, i] = p(x)
        dp = p.deriv()
        dA[:, i] = dp(x)
    return A, dA


def vander2d(x, degree):
    x = np.asarray(x)
    nNodes = (degree+1)*(degree+2)//2
    pq = pascal_triangle_monomials(degree)

    # It's easier to process if the input arrays
    # always have the same shape
    # If a 1D array is given (a single point),
    # convert to the equivalent 2D array
    x = x.reshape(-1,2)
    
    # switch to bi-unit triangle (-1,-1)--(1,-1)--(-1,1)
    z = 2.0*x - 1.0
    
    def map_from_tri_to_square(xi):
        small = 1e-12
        # The mapping has a singularity at the vertex (-1, 1).
        # Handle that point specially.
        indexSingular = xi[:, 1] > 1.0 - small
        xiShifted = xi.copy()
        xiShifted[indexSingular, 1] = 1.0 - small
        eta = np.zeros_like(xi)
        eta[:, 0] = 2.0*(1.0 + xiShifted[:, 0])/(1.0 - xiShifted[:, 1]) - 1.0
        eta[:, 1] = xiShifted[:, 1]
        eta[indexSingular, 0] = -1.0
        eta[indexSingular, 1] = 1.0
        
        # Jacobian of map. 
        # Actually, deta is just the first row of the Jacobian.
        # The second row is trivially [0, 1], so we don't compute it.
        # We just use that fact directly in the derivative Vandermonde
        # expressions.
        deta = np.zeros_like(xi)
        deta[:, 0] = 2/(1 - xiShifted[:, 1])
        deta[:, 1] = 2*(1 + xiShifted[:, 0])/(1 - xiShifted[:, 1])**2
        return eta, deta
    
    E, dE = map_from_tri_to_square(np.asarray(z))
    
    A = np.zeros((x.shape[0], nNodes))
    Ax = A.copy()
    Ay = A.copy()
    N1D = np.polynomial.Polynomial([0.5, -0.5])
    for i in range(nNodes):
        p = np.polynomial.Legendre.basis(pq[i, 0])
        
        # SciPy's polynomials use the deprecated poly1d type
        # of NumPy. To convert to the modern Polynomial type,
        # we need to reverse the order of the coefficients.
        qPoly1d = special.jacobi(pq[i, 1], 2*pq[i, 0] + 1, 0)
        q = np.polynomial.Polynomial(qPoly1d.coef[::-1])
        
        for j in range(pq[i, 0]):
            q *= N1D
        
        # orthonormality weight
        weight = np.sqrt((2*pq[i,0] + 1) * 2*(pq[i, 0] + pq[i, 1] + 1))
        
        A[:, i] = weight*p(E[:, 0])*q(E[:, 1])
        
        # derivatives
        dp = p.deriv()
        dq = q.deriv()
        Ax[:, i] = 2*weight*dp(E[:, 0])*q(E[:, 1])*dE[:, 0]
        Ay[:, i] = 2*weight*(dp(E[:, 0])*q(E[:, 1])*dE[:, 1]
                             + p(E[:, 0])*dq(E[:, 1]))
        
    return A, Ax, Ay


# TODO add hessians maybe?
class ShapeFunctions(NamedTuple):
    """
    Shape functions and shape function gradients (in the parametric space).

    :param values: Values of the shape functions at a discrete set of points.
        Shape is ``(nPts, nNodes)``, where ``nPts`` is the number of
        points at which the shame functinos are evaluated, and ``nNodes``
        is the number of nodes in the element (which is equal to the
        number of shape functions).
    :param gradients: Values of the parametric gradients of the shape functions.
        Shape is ``(nPts, nDim, nNodes)``, where ``nDim`` is the number
        of spatial dimensions. Line elements are an exception, which
        have shape ``(nPts, nNdodes)``.
    """
    values: Float[Array, "np nn"]
    gradients: Float[Array, "np nd nn"]


class BaseElement(eqx.Module):
    """
    Base class for different element technologies

    :param elementType: Element type name
    :param degree: Polynomial degree
    :param coordinates: Nodal coordinates in the reference configuration
    :param vertexNodes: Vertex node number, 0-based
    :param faceNodes: Nodes associated with each face, 0-based
    :param interiorNodes: Nodes in the interior, 0-based or empty
    """
    elementType: str
    degree: int
    coordinates: Float[Array, "nn nd"]
    vertexNodes: Int[Array, "nn"]
    faceNodes: Int[Array, "nf nnpf"]
    interiorNodes: Int[Array, "nni"]

    # def __init__(self, elementType, degree, coordinates, vertexNodes, faceNodes, interiorNodes):
    #     self.elementType = elementType
    #     self.degree = degree
    #     self.coordinates = coordinates
    #     self.vertexNodes = vertexNodes
    #     self.faceNodes = faceNodes
    #     self.interiorNodes = interiorNodes

    @abstractmethod
    def compute_shapes(self, nodalPoints, evaluationPoints):
        """
        Method to be defined to calculate shape function values
        and gradients given a list of nodal points (usually the vertexNodes)
        and a list of evaluation points (usually the quadrature points).
        """
        pass

    # TODO figure out how to rope in quadrature rules into this class