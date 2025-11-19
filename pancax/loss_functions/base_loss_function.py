from abc import abstractmethod
from jax import vmap
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp


class BaseLossFunction(eqx.Module):
    """
    Base class for loss functions.
    Currently does nothing but helps build a
    type hierarchy.
    """
    def filtered_loss(self, diff_params, static_params, *args, **kwargs):
        params = eqx.combine(diff_params, static_params)
        return self.__call__(params, *args, **kwargs)


class BCLossFunction(BaseLossFunction):
    """
    Base class for boundary condition loss functions.

    A ``load_step`` method is expect with the following
    type signature
    ``load_step(self, params, domain, t)``
    """

    @abstractmethod
    def load_step(self, params, domain, t):
        pass


class PhysicsLossFunction(BaseLossFunction):
    """
    Base class for physics loss functions.

    A ``load_step`` method is expect with the following
    type signature
    ``load_step(self, params, domain, t)``
    """
    @abstractmethod
    def load_step(self, params, domain, t):
        pass

    def path_dependent_loop(
        self, func, start, end, *args,
        use_fori_loop: Optional[bool] = False
    ):
        if use_fori_loop:
            def fori_loop_body(n, carry):
                return func(n, carry)

            return jax.lax.fori_loop(
                start, end, fori_loop_body, args
            )
        else:
            def scan_body(carry, n):
                return func(n, carry), None

            return jax.lax.scan(
                scan_body,
                args,
                jnp.arange(start, end)
            )[0]

    def state_variable_init(self, problem):
        ne = problem.domain.conns.shape[0]
        nq = len(problem.domain.fspace.quadrature_rule)

        def _vmap_func(n):
            return problem.physics.constitutive_model.\
                initial_state()

        if hasattr(problem.physics, "constitutive_model"):
            state_old = vmap(vmap(_vmap_func))(
                jnp.zeros((ne, nq))
            )
        else:
            state_old = jnp.zeros((ne, nq, 0))
        return state_old
