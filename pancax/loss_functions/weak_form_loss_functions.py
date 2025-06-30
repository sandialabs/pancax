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

    weight: float

    def __init__(self, weight: Optional[float] = 1.0):
        self.weight = weight

    def __call__(self, params, problem):
        dt = problem.times[1] - problem.times[0]
        energies = vmap(self.load_step, in_axes=(None, None, 0, None))(
            params, problem, problem.times, dt
        )
        energy = jnp.sum(energies)
        loss = energy
        return self.weight * loss, dict(energy=energy)

    def load_step(self, params, problem, t, dt):
        field, physics, state = params
        # hack for now, need a zero sized state var array
        state_old = jnp.zeros((
            problem.domain.conns.shape[0],
            problem.domain.fspace.num_quadrature_points, 0
        ))
        us = physics.vmap_field_values(field, problem.coords, t)
        pi, state_new = physics.potential_energy(
            physics, problem.domain, t, us, state_old, dt
        )
        return pi


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
    residual_weight: float

    def __init__(
        self,
        energy_weight: Optional[float] = 1.0,
        residual_weight: Optional[float] = 1.0,
    ):
        self.energy_weight = energy_weight
        self.residual_weight = residual_weight

    def __call__(self, params, problem):
        dt = problem.times[1] - problem.times[0]
        (pis, state_new), Rs = vmap(
            self.load_step,
            in_axes=(None, None, 0, None))(
            params, problem, problem.times, dt
        )
        # pi, R = jnp.sum(pis), jnp.sum(Rs)
        pi, R = jnp.sum(pis), jnp.mean(Rs)
        loss = self.energy_weight * pi + self.residual_weight * R
        return loss, dict(energy=pi, residual=R)

    def load_step(self, params, problem, t, dt):
        field, physics, state = params
        # hack for now, need a zero sized state var array
        state_old = jnp.zeros((
            problem.domain.conns.shape[0],
            problem.domain.fspace.num_quadrature_points, 0
        ))
        us = physics.vmap_field_values(field, problem.coords, t)
        return physics.potential_energy_and_residual(
            params, problem.domain, t, us, state_old, dt
        )


class EnergyResidualAndReactionLoss(PhysicsLossFunction):
    energy_weight: float
    residual_weight: float
    reaction_weight: float

    def __init__(
        self,
        energy_weight: Optional[float] = 1.0,
        residual_weight: Optional[float] = 1.0,
        reaction_weight: Optional[float] = 1.0,
    ):
        self.energy_weight = energy_weight
        self.residual_weight = residual_weight
        self.reaction_weight = reaction_weight

    def __call__(self, params, problem):
        dt = problem.times[1] - problem.times[0]
        (pis, states_new), Rs, reactions = vmap(
            self.load_step,
            in_axes=(None, None, 0, None))(
            params, problem, problem.times, dt
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

    def load_step(self, params, problem, t, dt):
        # field_network, props = params
        field, physics, state = params
        # us = domain.field_values(field_network, t)
        state_old = jnp.zeros((
            problem.domain.conns.shape[0],
            problem.domain.fspace.num_quadrature_points, 0
        ))
        us = physics.vmap_field_values(field, problem.coords, t)
        return physics.potential_energy_residual_and_reaction_force(
            params, problem.domain, t, us, state_old, dt,
            problem.global_data
        )


class PathDependentEnergyLoss(PhysicsLossFunction):
    weight: float

    def __init__(self, weight: Optional[float] = 1.0) -> None:
        self.weight = weight

    def __call__old(self, params, problem):
        field, physics, state = params

        ne = problem.domain.conns.shape[0]
        nq = len(problem.domain.fspace.quadrature_rule)

        def _vmap_func(n):
            return problem.physics.constitutive_model.\
                initial_state()

        state_old = vmap(vmap(_vmap_func))(
            jnp.zeros((ne, nq))
        )

        # TODO not adaptive
        # dumb implementation below
        dt = problem.times[1] - problem.times[0]
        pi = 0.0
        for n in range(problem.times.shape[0]):
            t = problem.times[n]
            pi_t, state_new = self.load_step(params, problem, t, dt, state_old)

            state_old = state_new
            pi = pi + pi_t

        loss = pi
        return self.weight * loss, dict(energy=pi)

    def __call__(self, params, problem):

        ne = problem.domain.conns.shape[0]
        nq = len(problem.domain.fspace.quadrature_rule)

        def _vmap_func(n):
            return problem.physics.constitutive_model.\
                initial_state()

        dt = problem.times[1] - problem.times[0]
        pi = 0.0
        state_old = vmap(vmap(_vmap_func))(
            jnp.zeros((ne, nq))
        )

        def body(n, carry):
            pi, state_old, dt = carry
            t = problem.times[n]
            pi_t, state_new = self.load_step(params, problem, t, dt, state_old)
            pi = pi + pi_t
            state_old = state_new
            dt = problem.times[n] - problem.times[n - 1]
            carry = pi, state_old, dt
            return carry

        # pi, state_old = jax.lax.fori_loop(
        #     0, len(problem.times), body, (pi, state_old)
        # )
        pi, state_old, dt = jax.lax.fori_loop(
            1, len(problem.times), body, (pi, state_old, dt)
        )
        loss = pi
        return self.weight * loss, dict(energy=pi)

    def load_step(self, params, problem, t, dt, state_old):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        pi, state_new = physics.potential_energy(
            physics, problem.domain, t, us, state_old, dt
        )
        return pi, state_new


class PathDependentEnergyResidualAndReactionLoss(PhysicsLossFunction):
    energy_weight: float
    residual_weight: float
    reaction_weight: float

    def __init__(
        self,
        energy_weight: Optional[float] = 1.0,
        residual_weight: Optional[float] = 1.0,
        reaction_weight: Optional[float] = 1.0,
    ):
        self.energy_weight = energy_weight
        self.residual_weight = residual_weight
        self.reaction_weight = reaction_weight

    def __call__(self, params, problem):
        ne = problem.domain.conns.shape[0]
        nq = len(problem.domain.fspace.quadrature_rule)

        def _vmap_func(n):
            return problem.physics.constitutive_model.\
                initial_state()

        dt = problem.times[1] - problem.times[0]
        pi = 0.0
        R = 0.0
        reaction = 0.0
        state_old = vmap(vmap(_vmap_func))(
            jnp.zeros((ne, nq))
        )

        def body(n, carry):
            pi, state_old, dt, R, reaction = carry
            t = problem.times[n]
            (pi_t, state_new), R_t, reaction_t = \
                self.load_step(params, problem, t, dt, state_old)
            pi = pi + pi_t
            R = R + R_t
            reaction = reaction + \
                jnp.square(reaction - problem.global_data.outputs[n])
            state_old = state_new
            dt = problem.times[n] - problem.times[n - 1]
            carry = pi, state_old, dt, R, reaction
            return carry

        pi, state_old, dt, R, reaction = jax.lax.fori_loop(
            1, len(problem.times), body, (pi, state_old, dt, R, reaction)
        )
        loss = self.energy_weight * pi + \
            self.residual_weight * (R / len(problem.times)) + \
            self.reaction_weight * (reaction / len(problem.times))
        return loss, dict(
            energy=pi,
            residual=R / len(problem.times),
            global_data_loss=reaction / len(problem.times)
        )

    def load_step(self, params, problem, t, dt, state_old):
        field, physics, state = params
        us = physics.vmap_field_values(field, problem.coords, t)
        return physics.potential_energy_residual_and_reaction_force(
            params, problem.domain, t, us, state_old, dt,
            problem.global_data
        )
