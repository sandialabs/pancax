import pytest


K = 0.833
G = 0.3846


@pytest.fixture
def neohookean_1():
  from pancax import NeoHookean
  return NeoHookean(
    bulk_modulus=K, 
    shear_modulus=G
  )


@pytest.fixture
def neohookean_2():
  from pancax import BoundedProperty, NeoHookean
  import jax
  key = jax.random.key(0)
  return NeoHookean(
    bulk_modulus=BoundedProperty(K, K, key),
    shear_modulus=BoundedProperty(G, G, key)
  )


def simple_shear_test(model):
  from .utils import simple_shear
  import jax
  import jax.numpy as jnp
  theta = 0.
  state_old = jnp.zeros((100, 0))
  dt = 1.
  gammas = jnp.linspace(0.0, 1., 100)
  Fs = jax.vmap(simple_shear)(gammas)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  Js = jax.vmap(model.jacobian)(grad_us)
  I1_bars = jax.vmap(model.I1_bar)(grad_us)
  psis, _ = jax.vmap(model.energy, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas, _ = jax.vmap(model.cauchy_stress, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )

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
  from .utils import uniaxial_strain
  import jax
  import jax.numpy as jnp
  theta = 0.
  state_old = jnp.zeros((100, 0))
  dt = 1.
  lambdas = jnp.linspace(1., 4., 100)
  Fs = jax.vmap(uniaxial_strain)(lambdas)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  Js = jax.vmap(model.jacobian)(grad_us)
  I1_bars = jax.vmap(model.I1_bar)(grad_us)
  psis, _ = jax.vmap(model.energy, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas, _ = jax.vmap(model.cauchy_stress, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )

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
