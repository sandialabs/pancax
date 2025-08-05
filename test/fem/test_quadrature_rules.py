from pancax.fem import QuadratureRule
from pancax.fem.elements import *
from scipy.special import binom
import jax.numpy as jnp
import pytest


# integrate x^n y^m on unit triangle
def integrate_2D_monomial_on_triangle(n, m):
    p = n + m
    return 1.0 / ((p + 2) * (p + 1) * binom(p, n))


def is_inside_hex(point):
    x_condition = (point[0] >= -1.0) and (point[0] <= 1.0)
    y_condition = (point[1] >= -1.0) and (point[1] <= 1.0)
    z_condition = (point[2] >= -1.0) and (point[2] <= 1.0)
    return x_condition and y_condition and z_condition


def is_inside_quad(point):
    x_condition = (point[0] >= -1.0) and (point[0] <= 1.0)
    y_condition = (point[1] >= -1.0) and (point[1] <= 1.0)
    return x_condition and y_condition


def is_inside_tet(point):
    x_condition = (point[0] >= 0.0) and (point[0] <= 1.0)
    y_condition = (point[1] >= 0.0) and (point[1] <= 1.0 - point[0])
    z_condition = (point[2] >= 0.0) and (point[2] <= 1.0 - point[0] - point[1])
    return x_condition and y_condition and z_condition


def is_inside_triangle(point):
    x_condition = (point[0] >= 0.0) and (point[0] <= 1.0)
    y_condition = (point[1] >= 0.0) and (point[1] <= 1.0 - point[0])
    return x_condition and y_condition


def is_inside_unit_interval(point):
    return jnp.all(point >= 0.0) and jnp.all(point <= 1.0)


elements_to_test = []
q_degrees = []
in_domain_methods = []
for q in range(1, 2 + 1):
    elements_to_test.append(Hex8Element())
    q_degrees.append(q)
    in_domain_methods.append(is_inside_hex)

for q in range(1, 25 + 1):
    elements_to_test.append(LineElement(1))
    q_degrees.append(q)
    in_domain_methods.append(is_inside_unit_interval)

for q in range(1, 3 + 1):
    elements_to_test.append(Quad4Element())
    elements_to_test.append(Quad9Element())
    q_degrees.append(q)
    q_degrees.append(q)
    in_domain_methods.append(is_inside_quad)
    in_domain_methods.append(is_inside_quad)

for q in range(1, 10 + 1):
    elements_to_test.append(SimplexTriElement(1))
    q_degrees.append(q)
    in_domain_methods.append(is_inside_unit_interval)

for q in range(1, 2 + 1):
    elements_to_test.append(Tet4Element())
    q_degrees.append(q)
    in_domain_methods.append(is_inside_tet)


for q in range(1, 2 + 1):
    elements_to_test.append(Tet10Element())
    q_degrees.append(q)
    in_domain_methods.append(is_inside_tet)


@pytest.mark.parametrize("el, q", zip(elements_to_test, q_degrees))
def test_are_postive_weights(el, q):
    if type(el) == Tet4Element and q == 2:
        pytest.skip("Not relevant for Tet4Element and q_degree = 2")
    if type(el) == Tet10Element and q == 2:
        pytest.skip("Not relevant for Tet10Element and q_degree = 2")
    qr = QuadratureRule(el, q)
    _, w = qr
    assert jnp.all(w > 0.0)


@pytest.mark.parametrize(
    "el, q, is_inside", zip(elements_to_test, q_degrees, in_domain_methods)
)
def test_are_inside_domain(el, q, is_inside):
    qr = QuadratureRule(el, q)
    for point in qr.xigauss:
        assert is_inside(point)


# TODO need general test method for other element formulations
def test_triangle_quadrature_exactness():
    max_degree = 10
    for degree in range(1, max_degree + 1):
        qr = QuadratureRule(SimplexTriElement(1), degree)
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                monomial = qr.xigauss[:, 0] ** i * qr.xigauss[:, 1] ** j
                quadratureAnswer = jnp.sum(monomial * qr.wgauss)
                exactAnswer = integrate_2D_monomial_on_triangle(i, j)
                assert jnp.abs(quadratureAnswer - exactAnswer) < 1e-14


def test_len_method():
    qr = QuadratureRule(Hex8Element(), 1)
    assert len(qr) == 1
    qr = QuadratureRule(Hex8Element(), 2)
    assert len(qr) == 8


# error checks
def test_error_raise_on_bad_element():
    with pytest.raises(TypeError):
        qr = QuadratureRule(dict(), 1)


def test_error_raise_on_bad_quadrature_degree():
    with pytest.raises(ValueError):
        qr = QuadratureRule(Hex8Element(), 3)

    with pytest.raises(ValueError):
        qr = QuadratureRule(Quad4Element(), 4)

    with pytest.raises(ValueError):
        qr = QuadratureRule(SimplexTriElement(1), 11)

    with pytest.raises(ValueError):
        qr = QuadratureRule(Tet4Element(), 3)
