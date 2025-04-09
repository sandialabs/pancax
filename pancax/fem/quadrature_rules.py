from .elements import Hex8Element
from .elements import LineElement
from .elements import Quad4Element, Quad9Element
from .elements import SimplexTriElement
from .elements import Tet4Element, Tet10Element
from .elements.base_element import BaseElement
from jax.lax import switch
from jaxtyping import Array, Float
from pancax.timer import Timer
import equinox as eqx
import jax.numpy as jnp
import math
import numpy as ojnp
import scipy.special


# TODO
# think about moving this stuff into elements


class QuadratureRule(eqx.Module):
    """
    Quadrature rule points and weights.
    A ``namedtuple`` containing ``xigauss``, a numpy array of the
    coordinates of the sample points in the reference domain, and
    ``wgauss``, a numpy array with the weights.

    :param xigauss: coordinates of gauss points in reference element
    :param wgauss: weights of gauss points in reference element
    """

    xigauss: Float[Array, "nq nd"]
    wgauss: Float[Array, "nq"]

    def __init__(self, parentElement: BaseElement, degree: int) -> None:
        with Timer("QuadratureRule.__init__"):
            if type(parentElement) is Hex8Element:
                xi, w = create_quadrature_rule_on_hex(degree)
            elif type(parentElement) is LineElement:
                xi, w = create_quadrature_rule_1D(degree)
            elif (
                type(parentElement) is Quad4Element
                or type(parentElement) is Quad9Element
            ):
                xi, w = create_quadrature_rule_on_quad(degree)
            elif type(parentElement) is SimplexTriElement:
                xi, w = create_quadrature_rule_on_triangle(degree)
            elif (
                type(parentElement) is Tet4Element
                or type(parentElement) is Tet10Element
            ):
                xi, w = create_quadrature_rule_on_tet(degree)
            else:
                raise TypeError(
                    f"Element type {type(parentElement)} does not "
                    "have a supported quadrature rule."
                )
            self.xigauss = xi
            self.wgauss = w

    # TODO maybe there's a better way?
    # TODO below is just so we don't have to change any tests for now
    def __iter__(self):
        yield self.xigauss
        yield self.wgauss

    def __len__(self):
        return self.xigauss.shape[0]

    def eval_at_iso_points(self, field):
        xigauss = self.xigauss
        fields = jnp.array(
            [field[0, :] + (field[1, :] - field[0, :]) * xi for xi in xigauss]
        )
        return fields


def create_quadrature_rule_1D(degree: int) -> QuadratureRule:
    """Creates a Gauss-Legendre quadrature on the unit interval.

    The rule can exactly integrate polynomials of degree up to
    ``degree``.

    Parameters
    ----------
    degree: Highest degree polynomial to be exactly integrated
        by the quadrature rule

    Returns
    -------
    A ``QuadratureRule`` named tuple containing the quadrature
    point coordinates
    and the weights.
    """

    n = math.ceil((degree + 1) / 2)
    xi, w = scipy.special.roots_sh_legendre(n)
    return xi, w


# TODO is this being used at all
def eval_at_iso_points(xigauss, field):
    fields = jnp.array(
        [field[0, :] + (field[1, :] - field[0, :]) * xi for xi in xigauss]
    )
    return fields


def create_quadrature_rule_on_hex(quad_degree: int) -> QuadratureRule:
    if quad_degree == 1:
        xi = jnp.array([[0.0, 0.0, 0.0]])
        w = 8.0 * jnp.ones(1)
    elif quad_degree == 2:
        xi = jnp.sqrt(1.0 / 3.0) * jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
        )
        w = jnp.ones(8)
    else:
        raise ValueError(
            f"Unsupported quadrature degree {quad_degree} on hex element."
        )

    return xi, w


def create_quadrature_rule_on_quad(quad_degree: int) -> QuadratureRule:
    """
    :param quad_degree: degree of quadrature rule to create
    """
    if quad_degree == 1:
        xi = jnp.array([[0.0, 0.0]])
        w = 4.0 * jnp.ones(1)
    elif quad_degree == 2:
        xi = jnp.sqrt(1.0 / 3.0) * jnp.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        )
        w = jnp.ones(4)
    elif quad_degree == 3:
        # TODO not quite accurate currently for some reason?
        # TODO
        # TODO
        xi = jnp.array(
            [
                [-jnp.sqrt(3.0 / 5.0), -jnp.sqrt(3.0 / 5.0)],
                [0.0, -jnp.sqrt(3.0 / 5.0)],
                [jnp.sqrt(3.0 / 5.0), -jnp.sqrt(3.0 / 5.0)],
                [-jnp.sqrt(3.0 / 5.0), 0.0],
                [0.0, 0.0],
                [jnp.sqrt(3.0 / 5.0), 0.0],
                [-jnp.sqrt(3.0 / 5.0), jnp.sqrt(3.0 / 5.0)],
                [0.0, jnp.sqrt(3.0 / 5.0)],
                [jnp.sqrt(3.0 / 5.0), jnp.sqrt(3.0 / 5.0)],
            ]
        )
        w = jnp.array(
            [
                25.0 / 81.0,
                40.0 / 81.0,
                25.0 / 81.0,
                40.0 / 81.0,
                64.0 / 81.0,
                40.0 / 81.0,
                25.0 / 81.0,
                40.0 / 81.0,
                25.0 / 80.0,
            ]
        )
    else:
        raise ValueError("Invalid quadrature degree: %s" % quad_degree)

    return xi, w


def create_quadrature_rule_on_tet(degree: int) -> QuadratureRule:
    if degree == 1:
        xi = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]])
        w = jnp.array([1.0 / 6.0])
    elif degree == 2:
        xi = jnp.array(
            [
                [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
                [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0],
                [1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0],
                [1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0],
            ]
        )
        w = jnp.array([
            -2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0
        ])
    else:
        raise ValueError("Invalid quadrature degree: %s" % degree)

    return xi, w


def create_quadrature_rule_on_triangle(degree: int) -> QuadratureRule:
    """Creates a Gauss-Legendre quadrature on the unit triangle.

    The rule can exactly integrate 2D polynomials up to the value of
    ``degree``. The domain is the triangle between the vertices
    (0, 0)-(1, 0)-(0, 1). The rules here are guaranteed to be
    cyclically symmetric in triangular coordinates and to have strictly
    positive weights.

    Parameters
    ----------
    degree: Highest degree polynomial to be exactly integrated by
        the quadrature rule

    Returns
    -------
    A ``QuadratureRule`` named tuple containing the
        quadrature point coordinates
    and the weights.
    """
    if degree == 1:
        xi = ojnp.array([[3.33333333333333333e-01, 3.33333333333333333e-01]])

        w = ojnp.array([5.00000000000000000e-01])
    elif degree == 2:
        xi = ojnp.array(
            [
                [6.66666666666666667e-01, 1.66666666666666667e-01],
                [1.66666666666666667e-01, 6.66666666666666667e-01],
                [1.66666666666666667e-01, 1.66666666666666667e-01],
            ]
        )

        w = ojnp.array(
            [
                1.66666666666666666e-01,
                1.66666666666666667e-01,
                1.66666666666666667e-01
            ]
        )
    elif degree <= 4:
        xi = ojnp.array(
            [
                [1.081030181680700e-01, 4.459484909159650e-01],
                [4.459484909159650e-01, 1.081030181680700e-01],
                [4.459484909159650e-01, 4.459484909159650e-01],
                [8.168475729804590e-01, 9.157621350977100e-02],
                [9.157621350977100e-02, 8.168475729804590e-01],
                [9.157621350977100e-02, 9.157621350977100e-02],
            ]
        )

        w = ojnp.array(
            [
                1.116907948390055e-01,
                1.116907948390055e-01,
                1.116907948390055e-01,
                5.497587182766100e-02,
                5.497587182766100e-02,
                5.497587182766100e-02,
            ]
        )
    elif degree <= 5:
        xi = ojnp.array(
            [
                [3.33333333333333e-01, 3.33333333333333e-01],
                [5.97158717897700e-02, 4.70142064105115e-01],
                [4.70142064105115e-01, 5.97158717897700e-02],
                [4.70142064105115e-01, 4.70142064105115e-01],
                [7.97426985353087e-01, 1.01286507323456e-01],
                [1.01286507323456e-01, 7.97426985353087e-01],
                [1.01286507323456e-01, 1.01286507323456e-01],
            ]
        )

        w = ojnp.array(
            [
                1.12500000000000e-01,
                6.61970763942530e-02,
                6.61970763942530e-02,
                6.61970763942530e-02,
                6.29695902724135e-02,
                6.29695902724135e-02,
                6.29695902724135e-02,
            ]
        )
    elif degree <= 6:
        xi = ojnp.array(
            [
                [5.01426509658179e-01, 2.49286745170910e-01],
                [2.49286745170910e-01, 5.01426509658179e-01],
                [2.49286745170910e-01, 2.49286745170910e-01],
                [8.73821971016996e-01, 6.30890144915020e-02],
                [6.30890144915020e-02, 8.73821971016996e-01],
                [6.30890144915020e-02, 6.30890144915020e-02],
                [5.31450498448170e-02, 3.10352451033784e-01],
                [6.36502499121399e-01, 5.31450498448170e-02],
                [3.10352451033784e-01, 6.36502499121399e-01],
                [5.31450498448170e-02, 6.36502499121399e-01],
                [6.36502499121399e-01, 3.10352451033784e-01],
                [3.10352451033784e-01, 5.31450498448170e-02],
            ]
        )

        w = ojnp.array(
            [
                5.83931378631895e-02,
                5.83931378631895e-02,
                5.83931378631895e-02,
                2.54224531851035e-02,
                2.54224531851035e-02,
                2.54224531851035e-02,
                4.14255378091870e-02,
                4.14255378091870e-02,
                4.14255378091870e-02,
                4.14255378091870e-02,
                4.14255378091870e-02,
                4.14255378091870e-02,
            ]
        )
    elif degree <= 10:
        xi = ojnp.array(
            [
                [0.33333333333333333e00, 0.33333333333333333e00],
                [0.4269134091050342e-02, 0.49786543295447483e00],
                [0.49786543295447483e00, 0.4269134091050342e-02],
                [0.49786543295447483e00, 0.49786543295447483e00],
                [0.14397510054188759e00, 0.42801244972905617e00],
                [0.42801244972905617e00, 0.14397510054188759e00],
                [0.42801244972905617e00, 0.42801244972905617e00],
                [0.6304871745135507e00, 0.18475641274322457e00],
                [0.18475641274322457e00, 0.6304871745135507e00],
                [0.18475641274322457e00, 0.18475641274322457e00],
                [0.9590375628566448e00, 0.20481218571677562e-01],
                [0.20481218571677562e-01, 0.9590375628566448e00],
                [0.20481218571677562e-01, 0.20481218571677562e-01],
                [0.3500298989727196e-01, 0.1365735762560334e00],
                [0.1365735762560334e00, 0.8284234338466947e00],
                [0.8284234338466947e00, 0.3500298989727196e-01],
                [0.1365735762560334e00, 0.3500298989727196e-01],
                [0.8284234338466947e00, 0.1365735762560334e00],
                [0.3500298989727196e-01, 0.8284234338466947e00],
                [0.37549070258442674e-01, 0.3327436005886386e00],
                [0.3327436005886386e00, 0.6297073291529187e00],
                [0.6297073291529187e00, 0.37549070258442674e-01],
                [0.3327436005886386e00, 0.37549070258442674e-01],
                [0.6297073291529187e00, 0.3327436005886386e00],
                [0.37549070258442674e-01, 0.6297073291529187e00],
            ]
        )

        w = ojnp.array(
            [
                0.4176169990259819e-01,
                0.36149252960283717e-02,
                0.36149252960283717e-02,
                0.36149252960283717e-02,
                0.3724608896049025e-01,
                0.3724608896049025e-01,
                0.3724608896049025e-01,
                0.39323236701554264e-01,
                0.39323236701554264e-01,
                0.39323236701554264e-01,
                0.3464161543553752e-02,
                0.3464161543553752e-02,
                0.3464161543553752e-02,
                0.147591601673897e-01,
                0.147591601673897e-01,
                0.147591601673897e-01,
                0.147591601673897e-01,
                0.147591601673897e-01,
                0.147591601673897e-01,
                0.1978968359803062e-01,
                0.1978968359803062e-01,
                0.1978968359803062e-01,
                0.1978968359803062e-01,
                0.1978968359803062e-01,
                0.1978968359803062e-01,
            ]
        )
    else:
        raise ValueError(
            "Quadrature of precision this high is not implemented."
        )

    return xi, w


# TODO remove this stuff
def create_padded_quadrature_rule_1D(degree):
    """Creates 1D Gauss quadrature rule data that are padded to maintain a
    uniform size, which makes this function jit-able.

    This function is inteded to be used only when jit
    compilation of calls to the
    quadrature rules are needed. Otherwise, prefer to
    use the standard quadrature
    rules. The standard rules do not contain extra 0s for padding, which makes
    them more efficient when  used repeatedly (such as in the global energy).

    Args:
      degree: degree of highest polynomial to be integrated exactly
    """

    jnpts = jnp.ceil((degree + 1) / 2).astype(int)
    xi, w = switch(
        jnpts,
        [
            _gauss_quad_1D_1pt,
            _gauss_quad_1D_2pt,
            _gauss_quad_1D_3pt,
            _gauss_quad_1D_4pt,
            _gauss_quad_1D_5pt,
        ],
        None,
    )
    return 0.5 * (xi + 1.0), 0.5 * w


def _gauss_quad_1D_1pt(_):
    xi = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    w = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0])
    return xi, w


def _gauss_quad_1D_2pt(_):
    xi = jnp.array([-0.5773502691896257, 0.5773502691896257, 0.0, 0.0, 0.0])
    w = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    return xi, w


def _gauss_quad_1D_3pt(_):
    xi = jnp.array([-0.7745966692414834, 0.0, 0.7745966692414834, 0.0, 0.0])
    w = jnp.array(
        [0.5555555555555557, 0.8888888888888888, 0.5555555555555557, 0.0, 0.0]
    )
    return xi, w


def _gauss_quad_1D_4pt(_):
    xi = jnp.array(
        [
            -0.8611363115940526,
            -0.33998104358485626,
            0.33998104358485626,
            0.8611363115940526,
            0.0,
        ]
    )
    w = jnp.array(
        [
            0.3478548451374537,
            0.6521451548625462,
            0.6521451548625462,
            0.3478548451374537,
            0.0,
        ]
    )
    return xi, w


def _gauss_quad_1D_5pt(_):
    xi = jnp.array(
        [
            -0.906179845938664,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.906179845938664,
        ]
    )
    w = jnp.array(
        [
            0.23692688505618942,
            0.4786286704993662,
            0.568888888888889,
            0.4786286704993662,
            0.23692688505618942,
        ]
    )

    return xi, w
