from pancax import BoundedProperty, NeoHookean
from .utils import *
import jax
import jax.numpy as jnp
import pytest


K = 0.833
G = 0.3846
key = jax.random.key(0)


@pytest.fixture
def neohookean_1():
  return NeoHookean(
    bulk_modulus=K, 
    shear_modulus=G
  )


@pytest.fixture
def neohookean_2():
  return NeoHookean(
    bulk_modulus=BoundedProperty(K, K, key),
    shear_modulus=BoundedProperty(G, G, key)
  )


def simple_shear_test(model):
  gammas = jnp.linspace(0.0, 1., 100)
  Fs = jax.vmap(simple_shear)(gammas)
  Js = jax.vmap(model.jacobian)(Fs)
  I1_bars = jax.vmap(model.I1_bar)(Fs)
  psis = jax.vmap(model.energy, in_axes=(0,))(Fs)#, props())
  sigmas = jax.vmap(model.cauchy_stress, in_axes=(0,))(Fs)#, props())

  for (psi, sigma, gamma, I1_bar, J) in zip(psis, sigmas, gammas, I1_bars, Js):
    psi_an = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J)) + \
             0.5 * G * (I1_bar - 3.)
    sigma_11_an = 2. / 3. * G * gamma**2
    sigma_22_an = -1. / 3. * G * gamma**2
    sigma_12_an = G * gamma
    assert jnp.allclose(psi, psi_an)
    assert jnp.allclose(sigma[0, 0], sigma_11_an)
    assert jnp.allclose(sigma[1, 1], sigma_22_an)
    assert jnp.allclose(sigma[2, 2], sigma_22_an)
    # #
    assert jnp.allclose(sigma[0, 1], sigma_12_an)
    assert jnp.allclose(sigma[1, 2], 0.0)
    assert jnp.allclose(sigma[2, 0], 0.0)
    # #
    assert jnp.allclose(sigma[1, 0], sigma_12_an)
    assert jnp.allclose(sigma[2, 1], 0.0)
    assert jnp.allclose(sigma[0, 2], 0.0)


def uniaxial_strain_test(model):
  lambdas = jnp.linspace(1., 4., 100)
  Fs = jax.vmap(uniaxial_strain)(lambdas)
  Js = jax.vmap(model.jacobian)(Fs)
  I1_bars = jax.vmap(model.I1_bar)(Fs)
  psis = jax.vmap(model.energy, in_axes=(0,))(Fs)#, props())
  sigmas = jax.vmap(model.cauchy_stress, in_axes=(0,))(Fs)#, props())
  # K, G = model.unpack_properties(props())

  for (psi, sigma, lambda_, I1_bar, J) in zip(psis, sigmas, lambdas, I1_bars, Js):
    psi_an = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J)) + \
              0.5 * G * (I1_bar - 3.)
    sigma_11_an = 0.5 * K * (lambda_ - 1. / lambda_) + \
                  2. / 3. * G * (lambda_**2 - 1.) * lambda_**(-5. / 3.)
    sigma_22_an = 0.5 * K * (lambda_ - 1. / lambda_) - \
                  1. / 3. * G * (lambda_**2 - 1.) * lambda_**(-5. / 3.)
    assert jnp.allclose(psi, psi_an)
    assert jnp.allclose(sigma[0, 0], sigma_11_an)
    assert jnp.allclose(sigma[1, 1], sigma_22_an)
    assert jnp.allclose(sigma[2, 2], sigma_22_an)
    #
    assert jnp.allclose(sigma[0, 1], 0.0)
    assert jnp.allclose(sigma[1, 2], 0.0)
    assert jnp.allclose(sigma[2, 0], 0.0)
    #
    assert jnp.allclose(sigma[1, 0], 0.0)
    assert jnp.allclose(sigma[2, 1], 0.0)
    assert jnp.allclose(sigma[0, 2], 0.0)


def test_simple_shear(neohookean_1, neohookean_2):
  simple_shear_test(neohookean_1)
  simple_shear_test(neohookean_2)


def test_uniaxial_strain(neohookean_1, neohookean_2):
  uniaxial_strain_test(neohookean_1)
  uniaxial_strain_test(neohookean_2)
