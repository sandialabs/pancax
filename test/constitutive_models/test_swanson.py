# from pancax import BoundedProperty, Swanson
# from .utils import *
# import jax 
# import jax.numpy as jnp
import pytest


K = 10.
A1 = 0.93074321
P1 = -0.07673672
B1 = 0.0
Q1 = -0.5
C1 = 0.10448312
R1 = 1.71691036


@pytest.fixture
def swanson_1():
  from pancax import Swanson
  return Swanson(
    bulk_modulus=K,
    A1=A1,
    P1=P1,
    B1=B1,
    Q1=Q1,
    C1=C1,
    R1=R1,
    cutoff_strain=0.01
  )


@pytest.fixture
def swanson_2():
  from pancax import BoundedProperty, Swanson
  import jax
  key = jax.random.key(0)
  return Swanson(
    bulk_modulus=BoundedProperty(K, K, key),
    A1=BoundedProperty(A1, A1, key),
    P1=BoundedProperty(P1, P1, key),
    B1=BoundedProperty(B1, B1, key),
    Q1=BoundedProperty(Q1, Q1, key),
    C1=BoundedProperty(C1, C1, key),
    R1=BoundedProperty(R1, R1, key),
    cutoff_strain=0.01
  )


def simple_shear_test(model):
  from .utils import simple_shear
  import jax
  import jax.numpy as jnp
  theta = 0.
  state_old = jnp.zeros((100, 0))
  dt = 1.
  gammas = jnp.linspace(0.05, 1., 100)
  Fs = jax.vmap(simple_shear)(gammas)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
  Js = jax.vmap(model.jacobian)(grad_us)
  I1_bars = jax.vmap(model.I1_bar)(grad_us)
  psis, _ = jax.vmap(model.energy, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas, _ = jax.vmap(model.cauchy_stress, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )

  def vmap_func(B_bar, I1_bar, J):
    psi_an = K * (J * jnp.log(J) - J + 1.) + \
              1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
              1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
    dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
                0.5 * C1 * (I1_bar / 3. - 1.)**R1
    B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
    sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)
    return psi_an, sigma_an[0, 0], sigma_an[1, 1], sigma_an[0, 1]

  psi_ans, sigma_11_ans, sigma_22_ans, sigma_12_ans = jax.vmap(
    vmap_func, in_axes=(0, 0, 0)
  )(B_bars, I1_bars, Js)

  assert jnp.allclose(psis, psi_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 0, 0], sigma_11_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 1, 1], sigma_22_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 2, 2], sigma_22_ans, atol=1e-3)
  #
  assert jnp.allclose(sigmas[:, 0, 1], sigma_12_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 1, 2], 0.0)
  assert jnp.allclose(sigmas[:, 2, 0], 0.0)
  # #
  assert jnp.allclose(sigmas[:, 1, 0], sigma_12_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 2, 1], 0.0)
  assert jnp.allclose(sigmas[:, 0, 2], 0.0)


def uniaxial_strain_test(model):
  from .utils import uniaxial_strain
  import jax
  import jax.numpy as jnp
  theta = 0.
  state_old = jnp.zeros((100, 0))
  dt = 1.
  lambdas = jnp.linspace(1.2, 4., 100)
  Fs = jax.vmap(uniaxial_strain)(lambdas)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
  Js = jax.vmap(model.jacobian)(grad_us)
  I1_bars = jax.vmap(model.I1_bar)(grad_us)
  psis, _ = jax.vmap(model.energy, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas, _ = jax.vmap(model.cauchy_stress, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )

  # for (psi, sigma, I1_bar, J, B_bar) in zip(psis, sigmas, I1_bars, Js, B_bars):
  def vmap_func(B_bar, I1_bar, J):
    psi_an = K * (J * jnp.log(J) - J + 1.) + \
              1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
              1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
    dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
                0.5 * C1 * (I1_bar / 3. - 1.)**R1
    B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
    sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)
    return psi_an, sigma_an[0, 0], sigma_an[1, 1]

  psi_ans, sigma_11_ans, sigma_22_ans = jax.vmap(
    vmap_func, in_axes=(0, 0, 0)
  )(B_bars, I1_bars, Js)

  assert jnp.allclose(psis, psi_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 0, 0], sigma_11_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 1, 1], sigma_22_ans, atol=1e-3)
  assert jnp.allclose(sigmas[:, 2, 2], sigma_22_ans, atol=1e-3)
  #
  assert jnp.allclose(sigmas[:, 0, 1], 0.0)
  assert jnp.allclose(sigmas[:, 1, 2], 0.0)
  assert jnp.allclose(sigmas[:, 2, 0], 0.0)
  #
  assert jnp.allclose(sigmas[:, 1, 0], 0.0)
  assert jnp.allclose(sigmas[:, 2, 1], 0.0)
  assert jnp.allclose(sigmas[:, 0, 2], 0.0)


def test_simple_shear(swanson_1, swanson_2):
  simple_shear_test(swanson_1)
  simple_shear_test(swanson_2)


def test_uniaxial_strain(swanson_1, swanson_2):
  uniaxial_strain_test(swanson_1)
  uniaxial_strain_test(swanson_2)
