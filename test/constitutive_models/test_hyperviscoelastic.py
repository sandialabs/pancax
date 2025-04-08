from pancax import *
from pancax.constitutive_models.mechanics.hyperviscoelasticity.base import *
from pancax.constitutive_models.mechanics.hyperviscoelasticity.simple_fefv import SimpleFeFv
from .utils import *
import jax
import jax.numpy as jnp
import matplotlib.pyplot
import pytest


@pytest.fixture
def model():
  return SimpleFeFv(
    NeoHookean(bulk_modulus=1000., shear_modulus=0.855),
    PronySeries(
      moduli=[1., 2., 3.], 
      relaxation_times=[1., 10., 100.]
    ),
    WLF(C1=17.44, C2=51.6, theta_ref=60.),
  )


def test_initial_state(model):
  state = model.initial_state()
  print(state)
  assert jnp.allclose(state, jnp.array([
    1., 0., 0., 0., 1., 0., 0., 0., 1.,
    1., 0., 0., 0., 1., 0., 0., 0., 1.,
    1., 0., 0., 0., 1., 0., 0., 0., 1.
  ]))


def test_extract_tensor(model):
  state = jnp.linspace(1., 27., 27)
  Fv_1 = model.extract_tensor(state, 0)
  Fv_2 = model.extract_tensor(state, 9)
  Fv_3 = model.extract_tensor(state, 18)

  assert jnp.allclose(Fv_1, jnp.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
  ]))
  assert jnp.allclose(Fv_2, jnp.array([
    [10., 11., 12.],
    [13., 14., 15.],
    [16., 17., 18.]
  ]))
  assert jnp.allclose(Fv_3, jnp.array([
    [19., 20., 21.],
    [22., 23., 24.],
    [25., 26., 27.]
  ]))

  Fvs = state.reshape((model.num_prony_terms(), 3, 3))

  assert jnp.allclose(Fvs[0, :, :], jnp.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
  ]))
  assert jnp.allclose(Fvs[1, :, :], jnp.array([
    [10., 11., 12.],
    [13., 14., 15.],
    [16., 17., 18.]
  ]))
  assert jnp.allclose(Fvs[2, :, :], jnp.array([
    [19., 20., 21.],
    [22., 23., 24.],
    [25., 26., 27.]
  ]))


def test_model(model):
  strain_rate = 1.e-2
  total_time = 100.0
  n_steps = 100
  times = jnp.linspace(0., total_time, n_steps)
  # F = uniaxial_strain(1.1)
  Fs = jax.vmap(lambda t: uniaxial_strain(jnp.exp(strain_rate * t)))(times)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  # grad_u = F - jnp.eye(3)
  theta = model.shift_factor_model.theta_ref
  state_old = model.initial_state()
  dt = total_time / n_steps

  # psi, state_new = model.energy(grad_u, theta, state_old, dt)
  # print(psi)
  # print(state_new)
  # print(model)

  energies = jnp.zeros(n_steps)
  states = jnp.zeros((n_steps, len(state_old)))

  energy_func = jax.jit(model.energy)
  log_strain_func = jax.jit(model.log_strain)

  for n, grad_u in enumerate(grad_us):
    psi, state_new = energy_func(grad_u, theta, state_old, dt)
    state_old = state_new
    energies = energies.at[n].set(psi)
    states = states.at[n, :].set(state_new)

  print(energies)
  print(states)

  # plt.figure(1)
  # plt.plot(times, states[:, 0])
  # plt.savefig('state_evolution.png')

  Gs = model.prony_series.moduli
  taus = model.prony_series.relaxation_times
  etas = Gs * taus

  for n in range(3):
    Fvs = jax.vmap(lambda Fv: Fv.at[9 * n:9 * (n + 1)].get().reshape((3, 3)))(states)
    Fes = jax.vmap(lambda F, Fv: F @ jnp.linalg.inv(Fv), in_axes=(0, 0))(Fs, Fvs)
    grad_uvs = jax.vmap(lambda F: F - jnp.eye(3))(Fvs)
    grad_ues = jax.vmap(lambda F: F - jnp.eye(3))(Fes)

    Evs = jax.vmap(log_strain_func)(grad_uvs)
    Ees = jax.vmap(log_strain_func)(grad_ues)

    # analytic solution
    e_v_11 = (2. / 3.) * strain_rate * times - \
             (2. / 3.) * strain_rate * taus[n] * (1. - jnp.exp(-times / taus[n]))

    e_e_11 = strain_rate * times - e_v_11
    e_e_22 = 0.5 * e_v_11

    # Ee_analytic = jax.vmap(
    #     lambda e_11, e_22: np.array(
    #         [[e_11, 0., 0.],
    #         [0., e_22, 0.],
    #         [0., 0., e_22]]
    #     ), in_axes=(0, 0)
    # )(e_e_11, e_e_22)

    # Me_analytic = jax.vmap(lambda Ee: 2. * Gs[n] * deviator(Ee))(Ee_analytic)
    # Dv_analytic = jax.vmap(lambda Me: 1. / (2. * self.etas[n]) * deviator(Me))(Me_analytic)
    # dissipated_energies_analytic += jax.vmap(lambda Dv: dt * self.etas[n] * np.tensordot(deviator(Dv), deviator(Dv)) )(Dv_analytic)

    # test
    assert jnp.isclose(Evs[:, 0, 0], e_v_11, atol=2.5e-3).all()
    assert jnp.isclose(Ees[:, 0, 0], e_e_11, atol=2.5e-3).all()
    assert jnp.isclose(Ees[:, 1, 1], e_e_22, atol=2.5e-3).all()


# NOTE this test is dumb... it's just checking vmap capabilities
def test_with_vmap(model):
  strain_rate = 1.e-2
  total_time = 100.0
  n_steps = 100
  times = jnp.linspace(0., total_time, n_steps)
  # F = uniaxial_strain(1.1)
  Fs = jax.vmap(lambda t: uniaxial_strain(jnp.exp(strain_rate * t)))(times)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  # grad_u = F - jnp.eye(3)
  theta = model.shift_factor_model.theta_ref
  state_old = model.initial_state()
  states_old = jnp.tile(state_old, (n_steps, 1))
  dt = total_time / n_steps

  energies = jnp.zeros(n_steps)
  states = jnp.zeros((n_steps, len(state_old)))

  energy_func = jax.jit(jax.vmap(model.energy, in_axes=(0, None, 0, None)))
  log_strain_func = jax.jit(model.log_strain)

  psis, states_new = energy_func(grad_us, theta, states_old, dt)
  print(psis.shape)
  print(states_new.shape)
  # assert False
