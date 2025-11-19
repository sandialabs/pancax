from .base_loss_function import PhysicsLossFunction
from jax import vmap
from typing import Optional
import jax
import jax.numpy as jnp


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

    def __call__(self, params, problem):
        if self.is_path_dependent:
            return self.path_dependent_call(params, problem)
        else:
            return self.path_independent_call(params, problem)

    def load_step(self, params, problem, t, dt, state_old):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        pi, state_new = physics.potential_energy(
            physics, problem.domain, t, us, state_old, dt
        )
        return pi, state_new

    def path_dependent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        pi = 0.0

        def body(n, carry):
            pi, state_old, dt = carry
            t = problem.times[n]
            pi_t, state_new = self.load_step(params, problem, t, dt, state_old)
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

    def path_independent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        energies, state_news = vmap(
            self.load_step,
            in_axes=(None, None, 0, None, None)
        )(
            params, problem, problem.times, dt, state_old
        )
        energy = jnp.sum(energies)
        loss = energy
        return self.weight * loss, dict(energy=energy)


class ResidualMSELoss(PhysicsLossFunction):
    weight: float

    def __init__(self, weight: Optional[float] = 1.0):
        self.weight = weight

    def __call__(self, params, domain):
        mses = vmap(self.load_step, in_axes=(None, None, 0))(
            params, domain, domain.times
        )
        mse = mses.mean()
        return self.weight * mse, dict(residual=mse)

    def load_step(self, params, domain, t):
        field, physics, state = params
        us = physics.vmap_field_values(field, domain.coords, t)
        rs = jnp.linalg.norm(physics.vmap_element_residual(
            field, domain, t, us
        ))
        return rs.mean()


class EnergyAndResidualLoss(PhysicsLossFunction):
    r"""
    Energy and residual loss function used in Hamel et. al

    Calculates the following quantity

    .. math::
      \mathcal{L} = w_1\Pi\left[u\right] + w_2\delta\Pi\left[u\right]_{free}

    :param energy_weight: Weight for the energy w_1
    :param residual_weight: Weight for the residual w_2
    """

    energy_weight: float
    is_path_dependent: bool
    residual_weight: float
    use_fori_loop: bool

    def __init__(
        self,
        energy_weight: Optional[float] = 1.0,
        is_path_dependent: Optional[bool] = False,
        residual_weight: Optional[float] = 1.0,
        use_fori_loop: Optional[bool] = False
    ):
        self.energy_weight = energy_weight
        self.is_path_dependent = is_path_dependent
        self.residual_weight = residual_weight
        self.use_fori_loop = use_fori_loop

    def __call__(self, params, problem):
        if self.is_path_dependent:
            assert False
        else:
            self.path_independent_call(params, problem)

    def path_dependent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        pi = 0.0
        R = 0.0

        def body(n, carry):
            pi, state_old, dt, R = carry
            t = problem.times[n]
            (pi_t, state_new), R_t = \
                self.load_step(params, problem, t, dt, state_old)
            pi = pi + pi_t
            R = R + R_t
            state_old = state_new
            dt = problem.times[n] - problem.times[n - 1]
            carry = pi, state_old, dt, R
            return carry

        if self.use_fori_loop:
            def fori_loop_body(n, carry):
                return body(n, carry)

            # starting at 1 assuming time step 0 is initial condition
            pi, state_old, dt, R = jax.lax.fori_loop(
                1, len(problem.times), fori_loop_body, (pi, state_old, dt, R)
            )
        else:
            def scan_body(carry, n):
                return body(n, carry), None

            (pi, state_old, dt, R), _ = jax.lax.scan(
                scan_body,
                (pi, state_old, dt, R),
                jnp.arange(1, len(problem.times))
            )

        loss = self.energy_weight * pi + \
            self.residual_weight * (R / len(problem.times))
        return loss, dict(
            energy=pi,
            residual=R / len(problem.times)
        )

    def path_independent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        (pis, state_new), Rs = vmap(
            self.load_step,
            in_axes=(None, None, 0, None, None))(
            params, problem, problem.times, dt, state_old
        )
        # pi, R = jnp.sum(pis), jnp.sum(Rs)
        pi, R = jnp.sum(pis), jnp.mean(Rs)
        loss = self.energy_weight * pi + self.residual_weight * R
        return loss, dict(energy=pi, residual=R)

    def load_step(self, params, problem, t, dt, state_old):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        return physics.potential_energy_and_residual(
            params, problem.domain, t, us, state_old, dt
        )


class EnergyResidualAndReactionLoss(PhysicsLossFunction):
    energy_weight: float
    is_path_dependent: bool
    residual_weight: float
    reaction_weight: float
    use_fori_loop: bool

    def __init__(
        self,
        energy_weight: Optional[float] = 1.0,
        is_path_dependent: Optional[bool] = False,
        residual_weight: Optional[float] = 1.0,
        reaction_weight: Optional[float] = 1.0,
        use_fori_loop: Optional[bool] = False
    ):
        self.energy_weight = energy_weight
        self.is_path_dependent = is_path_dependent
        self.residual_weight = residual_weight
        self.reaction_weight = reaction_weight
        self.use_fori_loop = use_fori_loop

    def __call__(self, params, problem):
        if self.is_path_dependent:
            return self.path_dependent_call(params, problem)
        else:
            return self.path_independent_call(params, problem)

    def path_dependent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        pi = 0.0
        R = 0.0
        reaction = 0.0

        def body(n, carry):
            pi, state_old, dt, R, reaction = carry
            t = problem.times[n]
            (pi_t, state_new), R_t, reaction_t = \
                self.load_step(params, problem, t, dt, state_old)
            pi = pi + pi_t
            R = R + R_t
            reaction = reaction + \
                jnp.square(reaction_t - problem.global_data.outputs[n])
            state_old = state_new
            dt = problem.times[n] - problem.times[n - 1]
            carry = pi, state_old, dt, R, reaction
            return carry

        pi, state_old, dt, R, reaction = self.path_dependent_loop(
            body, 1, len(problem.times),
            pi, state_old, dt, R, reaction,
            use_fori_loop=self.use_fori_loop
        )

        loss = self.energy_weight * pi + \
            self.residual_weight * (R / len(problem.times)) + \
            self.reaction_weight * (reaction / len(problem.times))
        return loss, dict(
            energy=pi,
            residual=R / len(problem.times),
            global_data_loss=reaction / len(problem.times)
        )

    def path_independent_call(self, params, problem):
        state_old = self.state_variable_init(problem)
        dt = problem.times[1] - problem.times[0]
        (pis, states_new), Rs, reactions = vmap(
            self.load_step,
            in_axes=(None, None, 0, None, None))(
            params, problem, problem.times, dt, state_old
        )
        pi, R = jnp.sum(pis), jnp.sum(Rs) / len(problem.times)
        reaction_loss = \
            jnp.square(reactions - problem.global_data.outputs).mean()
        loss = (
            self.energy_weight * pi
            + self.residual_weight * R
            + self.reaction_weight * reaction_loss
        )
        return loss, dict(
            energy=pi, residual=R,
            global_data_loss=reaction_loss, reactions=reactions
        )

    def load_step(self, params, problem, t, dt, state_old):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        return physics.potential_energy_residual_and_reaction_force(
            params, problem.domain, t, us, state_old, dt,
            problem.global_data
        )
