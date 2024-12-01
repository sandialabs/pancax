from pancax import BoundedProperty, Swanson
from .utils import *
import jax 
import jax.numpy as jnp
import pytest


K = 10.
A1 = 0.93074321
P1 = -0.07673672
B1 = 0.0
Q1 = -0.5
C1 = 0.10448312
R1 = 1.71691036
key = jax.random.key(0)


@pytest.fixture
def swanson_1():
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
  gammas = jnp.linspace(0.05, 1., 100)
  Fs = jax.vmap(simple_shear)(gammas)
  B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
  Js = jax.vmap(model.jacobian)(Fs)
  I1_bars = jax.vmap(model.I1_bar)(Fs)
  psis = jax.vmap(model.energy, in_axes=(0,))(Fs)
  sigmas = jax.vmap(model.cauchy_stress, in_axes=(0,))(Fs)

  for (psi, sigma, I1_bar, J, B_bar) in zip(psis, sigmas, I1_bars, Js, B_bars):
    psi_an = K * (J * jnp.log(J) - J + 1.) + \
              1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
              1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
    dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
                0.5 * C1 * (I1_bar / 3. - 1.)**R1
    B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
    sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)

    assert jnp.allclose(psi, psi_an, atol=1e-3)
    for i in range(3):
      for j in range(3):
        assert jnp.allclose(sigma[i, j], sigma_an[i, j], atol=1e-3)


def uniaxial_strain_test(model):
  lambdas = jnp.linspace(1.2, 4., 100)
  Fs = jax.vmap(uniaxial_strain)(lambdas)
  B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
  Js = jax.vmap(model.jacobian)(Fs)
  I1_bars = jax.vmap(model.I1_bar)(Fs)
  psis = jax.vmap(model.energy, in_axes=(0,))(Fs)
  sigmas = jax.vmap(model.cauchy_stress, in_axes=(0,))(Fs)

  for (psi, sigma, I1_bar, J, B_bar) in zip(psis, sigmas, I1_bars, Js, B_bars):
    psi_an = K * (J * jnp.log(J) - J + 1.) + \
              1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
              1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
    dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
                0.5 * C1 * (I1_bar / 3. - 1.)**R1
    B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
    sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)

    assert jnp.allclose(psi, psi_an, atol=1e-3)
    for i in range(3):
      for j in range(3):
        assert jnp.allclose(sigma[i, j], sigma_an[i, j], atol=1e-3)


def test_simple_shear(swanson_1, swanson_2):
  simple_shear_test(swanson_1)
  simple_shear_test(swanson_2)


def test_uniaxial_strain(swanson_1, swanson_2):
  uniaxial_strain_test(swanson_1)
  uniaxial_strain_test(swanson_2)
