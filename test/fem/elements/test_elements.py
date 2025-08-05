import pytest

tol = 1e-14


def check_1D_interpolant_in_element(element):
    import jax.numpy as jnp
    import numpy as onp
    p = element.coordinates
    return jnp.all(p >= 0.0) and onp.all(p <= 1.0)


def check_hex_interpolant_in_element(element):
    import jax.numpy as jnp
    p = element.coordinates
    return (
        jnp.all(p[:, 0] >= -1.0)
        and jnp.all(p[:, 0] <= 1.0)
        and jnp.all(p[:, 1] >= -1.0)
        and jnp.all(p[:, 1] <= 1.0)
        and jnp.all(p[:, 2] >= -1.0)
        and jnp.all(p[:, 2] <= 1.0)
    )


def check_quad_interpolant_in_element(element):
    import jax.numpy as jnp
    p = element.coordinates
    return (
        jnp.all(p[:, 0] >= -1.0)
        and jnp.all(p[:, 0] <= 1.0)
        and jnp.all(p[:, 1] >= -1.0)
        and jnp.all(p[:, 1] <= 1.0)
    )


def check_tet_interpolant_in_element(element):
    import jax.numpy as jnp
    p = element.coordinates
    # x conditions
    return (
        jnp.all(p[:, 0] >= -tol)
        and jnp.all(p[:, 0] <= 1.0 + tol)
        and jnp.all(p[:, 1] >= -tol)
        and jnp.all(p[:, 1] <= 1.0 - p[:, 0] + tol)
        and jnp.all(p[:, 2] >= -tol)
        and jnp.all(p[:, 2] <= 1.0 - p[:, 0] - p[:, 1] + tol)
    )


def check_tri_interpolant_in_element(element):
    import jax.numpy as jnp
    p = element.coordinates
    # x conditions
    return (
        jnp.all(p[:, 0] >= -tol)
        and jnp.all(p[:, 0] <= 1.0 + tol)
        and jnp.all(p[:, 1] >= -tol)
        and jnp.all(p[:, 1] <= 1.0 - p[:, 0] + tol)
    )


def generate_random_points_in_line(npts):
    import jax
    import numpy as onp
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    return onp.asarray(x)


def generate_random_points_in_hex(npts):
    import jax
    import numpy as onp
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.random.uniform(key, (npts,))
    z = jax.random.uniform(key, (npts,))
    points = jax.numpy.column_stack((x, y, z))
    return onp.asarray(points)


def generate_random_points_in_quad(npts):
    import jax
    import numpy as onp
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.random.uniform(key, (npts,))
    points = jax.numpy.column_stack((x, y))
    return onp.asarray(points)


def generate_random_points_in_tet(npts):
    import jax
    import numpy as onp
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.numpy.zeros(npts)
    z = jax.numpy.zeros(npts)
    for i in range(npts):
        key, subkey = jax.random.split(key)
        y = y.at[i].set(jax.random.uniform(
            subkey, minval=0.0, maxval=1.0 - x[i])
        )
        z = z.at[i].set(
            jax.random.uniform(subkey, minval=0.0, maxval=1.0 - x[i] - y[i])
        )
    points = jax.numpy.column_stack((x, y))
    return onp.asarray(points)


def generate_random_points_in_triangle(npts):
    import jax
    import numpy as onp
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.numpy.zeros(npts)
    for i in range(npts):
        key, subkey = jax.random.split(key)
        y = y.at[i].set(
            jax.random.uniform(subkey, minval=0.0, maxval=1.0 - x[i])
        )
    points = jax.numpy.column_stack((x, y))
    return onp.asarray(points)


@pytest.fixture
def elements_fix():
    from pancax.fem import QuadratureRule
    from pancax.fem.elements import Hex8Element
    from pancax.fem.elements import LineElement
    from pancax.fem.elements import Quad4Element
    from pancax.fem.elements import Quad9Element
    from pancax.fem.elements import SimplexTriElement
    from pancax.fem.elements import Tet4Element
    from pancax.fem.elements import Tet10Element

    q_rules = []
    elements = []
    check_interpolant_in_element_methods = []
    generate_random_points_method = []

    for q in range(1, 2 + 1):
        el = Hex8Element()
        q_rules.append(QuadratureRule(el, q))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_hex_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_hex)

    # Line elements
    for p in range(1, 25 + 1):
        el = LineElement(p)
        q_rules.append(QuadratureRule(el, p))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_1D_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_line)

    # Quad elements
    for q in range(1, 2 + 1):
        el = Quad4Element()
        q_rules.append(QuadratureRule(el, q))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_quad_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_quad)
        el = Quad9Element()
        q_rules.append(QuadratureRule(el, q))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_quad_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_quad)

    # Tet elements
    for q in range(1, 2 + 1):
        el = Tet4Element()
        q_rules.append(QuadratureRule(el, q))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_tet_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_tet)

    for q in range(1, 2 + 1):
        el = Tet10Element()
        q_rules.append(QuadratureRule(el, q))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_tet_interpolant_in_element
        )
        generate_random_points_method.append(generate_random_points_in_tet)

    # Tri elements
    for p in range(1, 6 + 1):
        el = SimplexTriElement(p)
        q_rules.append(QuadratureRule(el, p))
        elements.append(el)
        check_interpolant_in_element_methods.append(
            check_tri_interpolant_in_element
        )
        generate_random_points_method.append(
            generate_random_points_in_triangle
        )

    return \
        check_interpolant_in_element_methods, \
        elements, \
        generate_random_points_method, \
        q_rules


def test_interpolant_points_in_element(elements_fix):
    checks, els, _, _ = elements_fix
    for check, el in zip(checks, els):
        check(el)


# topology tests
# TODO generalize these methods below
def test_1D_element_element_topological_nodesets(elements_fix):
    from pancax.fem.elements import LineElement
    import jax.numpy as jnp
    for element in elements_fix:
        if type(element) is LineElement:
            p = element.coordinates
            jnp.isclose(p[element.vertexNodes[0]], 0.0)
            jnp.isclose(p[element.vertexNodes[1]], 1.0)

            if element.interiorNodes is not None:
                assert jnp.all(p[element.interiorNodes] > 0.0)
                assert jnp.all(p[element.interiorNodes] < 1.0)


def test_tri_element_element_topological_nodesets(elements_fix):
    from pancax.fem.elements import SimplexTriElement
    import jax.numpy as jnp
    import numpy as onp
    for element in elements_fix:
        if type(element) is SimplexTriElement:
            p = element.coordinates
            jnp.array_equal(
                p[element.vertexNodes[0], :], onp.array([1.0, 0.0])
            )
            jnp.array_equal(
                p[element.vertexNodes[1], :], onp.array([0.0, 1.0])
            )
            jnp.array_equal(
                p[element.vertexNodes[2], :], onp.array([0.0, 0.0])
            )

            if element.interiorNodes.size > 0:
                k = element.interiorNodes
                assert jnp.all(p[k, 0] > -tol)
                assert jnp.all(p[k, 1] + p[k, 0] - 1.0 < tol)


# TODO generalize these methods above
# TODO generalize these methods below
def test_tri_face_nodes_match_1D_lobatto_nodes(elements_fix):
    from pancax.fem.elements import LineElement, SimplexTriElement
    import jax.numpy as jnp
    elements1d = []
    elements2d = []
    for element in elements_fix:
        if type(element) is LineElement:
            elements1d.append(element)
        if type(element) is SimplexTriElement:
            elements2d.append(element)

    for element1d, elementTri in zip(elements1d, elements2d):
        for faceNodeIds in elementTri.faceNodes:
            # get the triangle face node points directly
            xf = elementTri.coordinates[faceNodeIds, :]
            # affine transformation of 1D node points to triangle face
            p = element1d.coordinates
            x1d = jnp.outer(1.0 - p, xf[0, :]) + jnp.outer(p, xf[-1, :])
            # make sure they are the same
            jnp.isclose(xf, x1d)


# TODO generalize these methods above
def test_partition_of_unity(elements_fix):
    from pancax.fem.elements import LineElement
    import jax.numpy as jnp
    _, els, _, qrs = elements_fix
    for el, qr in zip(els, qrs):
        if type(el) is LineElement:
            continue
        shapes, _ = el.compute_shapes(el.coordinates, qr.xigauss)
        assert jnp.allclose(jnp.sum(shapes, axis=1), jnp.ones(len(qr)))


def test_gradient_partition_of_unity(elements_fix):
    from pancax.fem.elements import LineElement
    import jax.numpy as jnp
    _, els, _, qrs = elements_fix
    for el, qr in zip(els, qrs):
        if type(el) is LineElement:
            continue
        _, shapeGradients = el.compute_shapes(el.coordinates, qr.xigauss)
        num_dim = qr.xigauss.shape[1]
        assert jnp.allclose(
            jnp.sum(shapeGradients, axis=1), jnp.zeros((len(qr), num_dim))
        )


def test_kronecker_delta_property(elements_fix):
    from pancax.fem.elements import LineElement
    import jax.numpy as jnp
    for _, el, _, _ in zip(*elements_fix):
        if type(el) is LineElement:
            continue
        shapeAtNodes, _ = el.compute_shapes(el.coordinates, el.coordinates)
        nNodes = el.coordinates.shape[0]
        assert jnp.allclose(shapeAtNodes, jnp.identity(nNodes))


# TODO generalize these methods below
def test_interpolation(elements_fix):
    from pancax.fem.elements import SimplexTriElement
    import jax.numpy as jnp
    import numpy as onp
    x = generate_random_points_in_triangle(1)
    for element in elements_fix:
        if type(element) is SimplexTriElement:
            degree = element.degree
            polyCoeffs = onp.fliplr(
                onp.triu(onp.ones((degree + 1, degree + 1)))
            )
            expected = onp.polynomial.polynomial.polyval2d(
                x[:, 0], x[:, 1], polyCoeffs
            )

            shape, _ = element.compute_shapes(element.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(
                element.coordinates[:, 0], element.coordinates[:, 1],
                polyCoeffs
            )
            fInterpolated = onp.dot(shape, fn)
            jnp.array_equal(expected, fInterpolated)


def test_grad_interpolation(elements_fix):
    from pancax.fem.elements import SimplexTriElement
    import jax.numpy as jnp
    import numpy as onp
    x = generate_random_points_in_triangle(1)
    for element in elements_fix:
        if type(element) is SimplexTriElement:
            degree = element.degree
            poly = onp.fliplr(onp.triu(onp.ones((degree + 1, degree + 1))))

            _, dShape = element.compute_shapes(element.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(
                element.coordinates[:, 0], element.coordinates[:, 1], poly
            )
            dfInterpolated = onp.einsum("qai,a->qi", dShape, fn)

            # exact x derivative
            direction = 0
            DPoly = onp.polynomial.polynomial.polyder(
                poly, 1, scl=1, axis=direction
            )
            expected0 = onp.polynomial.polynomial.polyval2d(
                x[:, 0], x[:, 1], DPoly
            )

            jnp.array_equal(expected0, dfInterpolated[:, 0])

            direction = 1
            DPoly = onp.polynomial.polynomial.polyder(
                poly, 1, scl=1, axis=direction
            )
            expected1 = onp.polynomial.polynomial.polyval2d(
                x[:, 0], x[:, 1], DPoly
            )

            jnp.array_equal(expected1, dfInterpolated[:, 1])


# TODO generalize these methods above
