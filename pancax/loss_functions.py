from abc import abstractmethod
from jax import vmap
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp


class BaseLossFunction(eqx.Module):
    # @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, None))
    @eqx.filter_vmap
    def filtered_ensemble_loss(
        self, diff_params, static_params, *args, **kwargs
    ):
        params = eqx.combine(diff_params, static_params)
        return self.__call__(params, *args, **kwargs)

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


class UserDefinedLossFunction(BaseLossFunction):
    func: any

    def __call__(self, params, problem, *args):
        return self.func(params, problem, *args)


class EnergyLoss(PhysicsLossFunction):
    r"""
    Energy loss function akin to the deep energy method.

    Calculates the following quantity

    .. math::
      \mathcal{L} = w\Pi\left[u\right] = \
        w\int_\Omega\psi\left(\mathbf{F}\right)

    :param weight: weight for this loss function
    """
    is_path_dependent: bool
    use_fori_loop: bool
    weight: float

    def __init__(
        self,
        is_path_dependent: Optional[bool] = False,
        use_fori_loop: Optional[bool] = False,  # for testing other looping
        weight: Optional[float] = 1.0
    ):
        self.is_path_dependent = is_path_dependent
        self.use_fori_loop = use_fori_loop
        self.weight = weight

    def __call__(self, params, problem, *args):
        if self.is_path_dependent:
            return self.path_dependent_call(params, problem, *args)
        else:
            return self.path_independent_call(params, problem, *args)

    def load_step(self, params, problem, t, dt, state_old, *args):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        pi, state_new = physics.potential_energy(
            physics, problem.domain, t, us, state_old, dt, *args
        )
        return pi, state_new

    def path_dependent_call(self, params, problem, *args):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        pi = 0.0

        def body(n, carry):
            pi, state_old, dt = carry
            t = problem.times[n]
            pi_t, state_new = self.load_step(
                params, problem, t, dt, state_old, *args
            )
            pi = pi + pi_t
            state_old = state_new
            dt = problem.times[n] - problem.times[n - 1]
            carry = pi, state_old, dt
            return carry

        pi, state_old, dt = self.path_dependent_loop(
            body, 1, len(problem.times),
            pi, state_old, dt,
            use_fori_loop=self.use_fori_loop
        )

        loss = pi
        return self.weight * loss, dict(energy=pi)

    def path_independent_call(self, params, problem, *args):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        in_axes = (None, None, 0, None, None) + len(args) * (None,)

        energies, state_news = vmap(
            self.load_step,
            # in_axes=(None, None, 0, None, None)
            in_axes=in_axes
        )(
            params, problem, problem.times, dt, state_old
        )
        energy = jnp.sum(energies)
        loss = energy
        return self.weight * loss, dict(energy=energy)


