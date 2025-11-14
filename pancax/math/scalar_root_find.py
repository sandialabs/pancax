# boosted from optimism
import equinox as eqx
import jax
import jax.numpy as jnp


class Settings(eqx.Module):
    max_iters: int
    x_tol: float
    r_tol: float


def get_settings(max_iters=50, x_tol=1e-13, r_tol=0):
    return Settings(max_iters, x_tol, r_tol)


def find_root(f, x0, bracket, settings):
    """Find a root of a nonlinear scalar-valued equation.

    Uses Newton's method, safeguarded with bisection.
    See rtsafe(...) from Numerical Recipes. This function is differentiable
    using Jax transforms, and can be compsed with `vmap` and `jit`.

    Parameters
    ==========
    f : callable
        Scalar function of which to find a root.
    x0 : real
        Initial guess for root. The value of x0 should be within the range
        defined by bracket. If not, the initial guess will be automatically
        clipped to the nearest bound.
    bracket : sequence of 2 reals (list, tuple, numpy array, etc)
        Upper and lower bounds for the root search. The existence of a
        root is inferred by checking that the values of `f` at both brackets
        do not have the same sign. If this check fails, `nan` is returned.
    settings : A settings object from this module
        Algorithmic settings.

    Returns
    =======
    x : real (or nan)
        Argument of f such that f(x) = 0 (within provided tolerance). If the
        root is not bracketed, `nan` is returned.
    """
    return jax.lax.custom_root(
        f, x0,
        lambda F, X0: rtsafe_(F, X0, bracket, settings),
        lambda g, y: y / g(1.0),
        has_aux=True
    )


def rtsafe_(f, x0, bracket, settings):
    # Find root of a scalar function
    # Newton's method, safeguarded with bisection
    # from Numerical Recipes

    max_iters = settings.max_iters
    x_tol = settings.x_tol
    r_tol = settings.r_tol

    f_and_fprime = jax.value_and_grad(f)

    converged = False

    fl = f(bracket[0])
    fh = f(bracket[1])
    functionCalls = 2

    # Fix initial guess if outside bracket.
    # Must do this before the steps that change the initial guess
    # as a means to exit early (ie, root bracketing check and
    # testing bracket values as solutions).
    x0 = jnp.clip(x0, bracket[0], bracket[1])

    # check that root is bracketed
    x0 = jnp.where(
        fl * fh < 0.0,
        x0,
        jnp.nan
    )

    # Check if either bracket is a root
    leftBracketIsSolution = (fl == 0.0)
    x0 = jnp.where(leftBracketIsSolution, bracket[0], x0)
    converged = jnp.where(leftBracketIsSolution, True, converged)

    rightBracketIsSolution = (fh == 0.0)
    x0 = jnp.where(rightBracketIsSolution, bracket[1], x0)
    converged = jnp.where(rightBracketIsSolution, True, converged)

    # ORIENT THE SEARCH SO THAT F(XL) < 0.
    xl, xh = jax.lax.cond(
        fl < 0,
        lambda b: (b[0], b[1]),
        lambda b: (b[1], b[0]),
        bracket
    )

    # INITIALIZE THE ''STEP SIZE BEFORE LAST'', AND THE LAST STEP
    dxOld = jnp.abs(bracket[1] - bracket[0])
    dx = dxOld

    F, DF = f_and_fprime(x0)
    functionCalls += 1

    def cond(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry
        keepLooping = (~converged) & (i < max_iters)
        return keepLooping

    def loop_body(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry

        newtonOutOfRange = (
            (root - xh) * DF - F) * \
            ((root - xl) * DF - F) > 0
        newtonDecreasingSlowly = jnp.abs(2. * F) > jnp.abs(dxOld * DF)
        dxOld = dx
        root, dx, converged = jax.lax.cond(
            newtonOutOfRange | newtonDecreasingSlowly,
            bisection_step,
            newton_step,
            root, xl, xh, DF, F
        )

        F, DF = f_and_fprime(root)

        # MAINTAIN THE BRACKET ON THE ROOT
        xl, xh = jax.lax.cond(
            F < 0,
            lambda rt, lo, hi: (rt, hi),
            lambda rt, lo, hi: (lo, rt),
            root, xl, xh
        )
        i += 1
        converged = converged | (jnp.abs(dx) < x_tol) | (jnp.abs(F) < r_tol)
        return root, dx, dxOld, F, DF, xl, xh, converged, i

    x, dx, _, F, _, _, _, converged, iters = jax.lax.while_loop(
        cond,
        loop_body,
        (x0, dx, dxOld, F, DF, xl, xh, converged, 0)
    )

    x = jnp.where(converged, x, jnp.nan)

    # BT 10/14/2025 As of Jax 0.4.34,
    # the has_aux argument of custom_root is broken
    # and cannot handle non-differentiable outputs.
    # See https://github.com/jax-ml/jax/issues/24295
    # return x, SolutionInfo(converged=converged, function_calls=functionCalls,
    #
    # iterations=iters, residual_norm=np.abs(F), correction_norm=np.abs(dx))
    return x, None


def bisection_step(x, xl, xh, df, f):
    dx = 0.5 * (xh - xl)
    x = xl + dx
    converged = (x == xl)
    return x, dx, converged


def newton_step(x, xl, xh, df, f):
    dx = -f / df
    temp = x
    x = x + dx
    converged = (x == temp)
    return x, dx, converged
