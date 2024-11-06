from pancax.fem import QuadratureRule
from pancax.fem.elements import *
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

tol = 1e-14


def check_1D_interpolant_in_element(element):
    p = element.coordinates
    return jnp.all(p >= 0.0) and onp.all(p <= 1.0)


def check_hex_interpolant_in_element(element):
    p = element.coordinates
    return jnp.all(p[:, 0] >= -1.0) and \
           jnp.all(p[:, 0] <= 1.0) and \
           jnp.all(p[:, 1] >= -1.0) and \
           jnp.all(p[:, 1] <= 1.0) and \
           jnp.all(p[:, 2] >= -1.0) and \
           jnp.all(p[:, 2] <= 1.0)


def check_quad_interpolant_in_element(element):
    p = element.coordinates
    return jnp.all(p[:, 0] >= -1.0) and \
           jnp.all(p[:, 0] <= 1.0) and \
           jnp.all(p[:, 1] >= -1.0) and \
           jnp.all(p[:, 1] <= 1.0)


def check_tet_interpolant_in_element(element):
    p = element.coordinates
    # x conditions
    return jnp.all(p[:, 0] >= -tol) and \
           jnp.all(p[:, 0] <= 1.0 + tol) and \
           jnp.all(p[:, 1] >= -tol) and \
           jnp.all(p[:, 1] <= 1. - p[:, 0] + tol) and \
           jnp.all(p[:, 2] >= -tol) and \
           jnp.all(p[:, 2] <= 1. - p[:, 0] - p[:, 1] + tol)


def check_tri_interpolant_in_element(element):
    p = element.coordinates
    # x conditions
    return jnp.all(p[:, 0] >= -tol) and \
           jnp.all(p[:, 0] <= 1.0 + tol) and \
           jnp.all(p[:, 1] >= -tol) and \
           jnp.all(p[:, 1] <= 1. - p[:,0] + tol)


def generate_random_points_in_line(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    return onp.asarray(x)


def generate_random_points_in_hex(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.random.uniform(key, (npts,))
    z = jax.random.uniform(key, (npts,))
    points = jax.numpy.column_stack((x, y, z))
    return  onp.asarray(points)


def generate_random_points_in_quad(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.random.uniform(key, (npts,))
    points = jax.numpy.column_stack((x, y))
    return  onp.asarray(points)


def generate_random_points_in_tet(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.numpy.zeros(npts)
    z = jax.numpy.zeros(npts)
    for i in range(npts):
        key,subkey = jax.random.split(key)
        y = y.at[i].set(jax.random.uniform(subkey, minval=0.0, maxval=1.0 - x[i]))
        z = z.at[i].set(jax.random.uniform(subkey, minval=0.0, maxval=1.0 - x[i] - y[i]))
    points = jax.numpy.column_stack((x, y))
    return  onp.asarray(points)


def generate_random_points_in_triangle(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.numpy.zeros(npts)
    for i in range(npts):
        key,subkey = jax.random.split(key)
        y = y.at[i].set(jax.random.uniform(subkey, minval=0.0, maxval=1.0-x[i]))
    points = jax.numpy.column_stack((x, y))
    return  onp.asarray(points)


q_rules = []
elements = []
check_interpolant_in_element_methods = []
generate_random_points_method = []
# Hex elements
for q in range(1, 2 + 1):
    el = Hex8Element()
    q_rules.append(QuadratureRule(el, q))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_hex_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_hex)

# Line elements
for p in range(1, 25 + 1):
    el = LineElement(p)
    q_rules.append(QuadratureRule(el, p))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_1D_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_line)

# Quad elements
for q in range(1, 2 + 1):
    el = Quad4Element()
    q_rules.append(QuadratureRule(el, q))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_quad_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_quad)
    el = Quad9Element()
    q_rules.append(QuadratureRule(el, q))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_quad_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_quad)

for q in range(1, 2 + 1):
    el = Tet4Element()
    q_rules.append(QuadratureRule(el, q))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_tet_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_tet)


for q in range(1, 2 + 1):
    el = Tet10Element()
    q_rules.append(QuadratureRule(el, q))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_tet_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_tet)


# Tri elements
for p in range(1, 6 + 1):
    el = SimplexTriElement(p)
    q_rules.append(QuadratureRule(el, p))
    elements.append(el)
    check_interpolant_in_element_methods.append(check_tri_interpolant_in_element)
    generate_random_points_method.append(generate_random_points_in_triangle)


@pytest.mark.parametrize('el, check', zip(elements, check_interpolant_in_element_methods))
def test_interpolant_points_in_element(el, check):
    assert check(el)


# topology tests
# TODO generalize these methods below
def test_1D_element_element_topological_nodesets():
    for element in elements:
        if type(element) == LineElement:
            p = element.coordinates
            jnp.isclose(p[element.vertexNodes[0]], 0.0)
            jnp.isclose(p[element.vertexNodes[1]], 1.0)
            
            if element.interiorNodes is not None:
                assert jnp.all(p[element.interiorNodes] > 0.0)
                assert jnp.all(p[element.interiorNodes] < 1.0)


def test_tri_element_element_topological_nodesets():
    for element in elements:
        if type(element) == SimplexTriElement:
            p = element.coordinates
            jnp.array_equal(p[element.vertexNodes[0], :], onp.array([1.0, 0.0]))
            jnp.array_equal(p[element.vertexNodes[1], :], onp.array([0.0, 1.0]))
            jnp.array_equal(p[element.vertexNodes[2], :], onp.array([0.0, 0.0]))
            
            if element.interiorNodes.size > 0:
                k = element.interiorNodes
                assert jnp.all(p[k,0] > -tol)
                assert jnp.all(p[k,1] + p[k,0] - 1. <  tol)
# TODO generalize these methods above
# TODO generalize these methods below
def test_tri_face_nodes_match_1D_lobatto_nodes():
    elements1d = []
    elements2d = []
    for element in elements:
        if type(element) == LineElement:
            elements1d.append(element)
        if type(element) == SimplexTriElement:
            elements2d.append(element)
    
    for element1d, elementTri in zip(elements1d, elements2d):
        for faceNodeIds in elementTri.faceNodes:
            # get the triangle face node points directly
            xf = elementTri.coordinates[faceNodeIds,:]
            # affine transformation of 1D node points to triangle face
            p = element1d.coordinates
            x1d = jnp.outer(1.0 - p, xf[0,:]) + jnp.outer(p, xf[-1,:])
            # make sure they are the same
            jnp.isclose(xf, x1d)

# TODO generalize these methods above
@pytest.mark.parametrize('el, qr', zip(elements, q_rules))
def test_partition_of_unity(el, qr):
    if type(el) == LineElement:
        pytest.skip('LineElement failing for now')
    shapes, _ = el.compute_shapes(el.coordinates, qr.xigauss)
    assert jnp.allclose(jnp.sum(shapes, axis=1), jnp.ones(len(qr)))


@pytest.mark.parametrize('el, qr', zip(elements, q_rules))
def test_gradient_partition_of_unity(el, qr):
    if type(el) == LineElement:
        pytest.skip('LineElement failing for now')
    _, shapeGradients = el.compute_shapes(el.coordinates, qr.xigauss)
    num_dim = qr.xigauss.shape[1]
    assert jnp.allclose(jnp.sum(shapeGradients, axis=1), jnp.zeros((len(qr), num_dim)))


@pytest.mark.parametrize('el', elements)
def test_kronecker_delta_property(el):
    if type(el) == LineElement:
        pytest.skip('LineElement failing for now')
    shapeAtNodes, _ = el.compute_shapes(el.coordinates, el.coordinates)
    nNodes = el.coordinates.shape[0]
    assert jnp.allclose(shapeAtNodes, jnp.identity(nNodes))


# TODO generalize these methods below
def test_interpolation():
    x = generate_random_points_in_triangle(1)
    for element in elements:
        if type(element) == SimplexTriElement:
            degree = element.degree
            polyCoeffs = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))
            expected = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], polyCoeffs)

            shape, _ = element.compute_shapes(element.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                        element.coordinates[:,1],
                                                        polyCoeffs)
            fInterpolated = onp.dot(shape, fn)
            jnp.array_equal(expected, fInterpolated)


def test_grad_interpolation():
    x = generate_random_points_in_triangle(1)
    for element in elements:
        if type(element) == SimplexTriElement:
            degree = element.degree
            poly = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))

            _, dShape = element.compute_shapes(element.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                        element.coordinates[:,1],
                                                        poly)
            dfInterpolated = onp.einsum('qai,a->qi',dShape, fn)

            # exact x derivative
            direction = 0
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected0 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            jnp.array_equal(expected0, dfInterpolated[:,0])

            direction = 1
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected1 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            jnp.array_equal(expected1, dfInterpolated[:,1])
# TODO generalize these methods above
