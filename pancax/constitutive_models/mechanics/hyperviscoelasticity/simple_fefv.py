from .base import Scalar, State, Tensor
from .base import HyperelasticModel, HyperViscoElastic, PronySeries, ShiftFactorModel
from ....math import tensor_math
import equinox as eqx
import jax
import jax.numpy as jnp


class SimpleFeFv(HyperViscoElastic):
  eq_model: HyperelasticModel
  prony_series: PronySeries = eqx.field(static=True)
  shift_factor_model: ShiftFactorModel = eqx.field(static=True)

  def dissipation(self, Dv, G, tau):
    eta = G * tau
    return eta * tensor_math.norm_of_deviator_squared(Dv)

  def energy(self, grad_u: Tensor, theta: Scalar, Z: State, dt: Scalar) -> Scalar:
    # setup properties
    a_T = self.shift_factor_model(grad_u, theta, Z, dt)
    Gs = self.prony_series.moduli
    taus = a_T * self.prony_series.relaxation_times

    # extract states
    n_prony = self.num_prony_terms()
    Fvs_old = Z.reshape((n_prony, 3, 3))

    # kinematics
    F = self.deformation_gradient(grad_u)
    Fe_trials = jax.vmap(lambda Fv_old: F @ jnp.linalg.inv(Fv_old))(Fvs_old)
    Ce_trials = jax.vmap(lambda Fe: Fe.T @ Fe)(Fe_trials)
    Ee_trials = jax.vmap(tensor_math.mtk_log_sqrt)(Ce_trials)

    # update state
    delta_Evs = jax.vmap(self.state_increment, in_axes=(0, 0, 0, None))(
      Ee_trials, Gs, taus, dt
    )
    Ees = Ee_trials - delta_Evs
    Dvs = delta_Evs / dt
    Fvs_new = jax.vmap(lambda Fv_old, delta_Ev: jax.scipy.linalg.expm(delta_Ev) @ Fv_old, in_axes=(0, 0))(
      Fvs_old, delta_Evs
    )
    Z = Fvs_new.flatten()

    # constitutive calculation (energy and dissipation)
    psi_eq = self.eq_model.energy(F)
    psi_neq = jnp.sum(jax.vmap(self.neq_strain_energy, in_axes=(0, 0))(
      Ees, Gs
    ))
    d = jnp.sum(jax.vmap(self.dissipation, in_axes=(0, 0, 0))(
      Dvs, Gs, taus
    ))
    return psi_eq + psi_neq + d, Z

  def initial_state(self):
    Fvs = jax.vmap(lambda _: jnp.eye(3))(range(self.num_prony_terms()))
    return Fvs.flatten()

  def neq_strain_energy(self, Ee, G):
    return G * tensor_math.norm_of_deviator_squared(Ee)

  # operates on a singel prony branch
  def state_increment(self, Ee_trial, G, tau, dt):
    Ee_trial_dev = tensor_math.dev(Ee_trial)
    integration_factor = 1. / (1. + dt / tau)
    return dt * integration_factor * Ee_trial_dev / tau
