import pytest


K = 0.833
G = 0.3846

@pytest.fixture
def hencky_1():
  from pancax import Hencky
  return Hencky(
    bulk_modulus=K, 
    shear_modulus=G
  )

@pytest.fixture
def hencky_2():
  from pancax import BoundedProperty, Hencky
  import jax
  key = jax.random.key(0)
  return Hencky(
    bulk_modulus=BoundedProperty(K, K, key), 
    shear_modulus=BoundedProperty(G, G, key)
  )


# TODO fix this test

def simple_shear_test(model):
  from pancax.math import tensor_math
  from .utils import simple_shear
  import jax
  import jax.numpy as jnp
  theta = 0.
  state_old = jnp.zeros((100, 0))
  dt = 1.
  gammas = jnp.linspace(0.0, 1., 100)
  Fs = jax.vmap(simple_shear)(gammas)
  grad_us = jax.vmap(lambda F: F - jnp.eye(3))(Fs)
  # Js = jax.vmap(model.jacobian)(grad_us)
  # I1_bars = jax.vmap(model.I1_bar)(grad_us)
  Js = jax.vmap(model.jacobian)(grad_us)
  Es = jax.vmap(model.log_strain)(grad_us)
  trEs = jax.vmap(jnp.trace)(Es)

  # Edevs = jax.vmap(lambda E: E - (1. / 3.) * jnp.trace(E) * jnp.eye(3))(Es)
  Edevs = jax.vmap(tensor_math.dev)(Es)
  psis, _ = jax.vmap(model.energy, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas, _ = jax.vmap(model.cauchy_stress, in_axes=(0, None, 0, None))(
    grad_us, theta, state_old, dt
  )
  sigmas_an = jax.vmap(
    lambda trE, devE, J: (K * trE * jnp.eye(3) + 2. * G * devE) / J, in_axes=(0, 0, 0)
  )(trEs, Edevs, Js)

  # print(psis)
  # print(Es[-1, :, :])
  # print(sigmas[-1, :, :])
  # print(sigmas_an[-1, :, :])
  # # print(sigmas - sigmas_an)
  # assert jnp.allclose(sigmas, sigmas_an, atol=1e-8)
  # assert False
  # for (psi, sigma, gamma, I1_bar, J) in zip(psis, sigmas, gammas, I1_bars, Js):
  #   psi_an = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J)) + \
  #             -0.5 * G * Jm * jnp.log(1. - (I1_bar - 3.) / Jm)
  #   sigma_11_an = 2. / 3. * G * Jm * gamma**2  / (Jm - gamma**2)
  #   sigma_22_an = -1. / 3. * G * Jm * gamma**2  / (Jm - gamma**2)
  #   sigma_12_an = G * Jm * gamma / (Jm - gamma**2)
  #   assert jnp.allclose(psi, psi_an)
  #   assert jnp.allclose(sigma[0, 0], sigma_11_an)
  #   assert jnp.allclose(sigma[1, 1], sigma_22_an)
  #   assert jnp.allclose(sigma[2, 2], sigma_22_an)
  #   # #
  #   assert jnp.allclose(sigma[0, 1], sigma_12_an)
  #   assert jnp.allclose(sigma[1, 2], 0.0)
  #   assert jnp.allclose(sigma[2, 0], 0.0)
  #   # #
  #   assert jnp.allclose(sigma[1, 0], sigma_12_an)
  #   assert jnp.allclose(sigma[2, 1], 0.0)
  #   assert jnp.allclose(sigma[0, 2], 0.0)


def test_simple_shear(hencky_1, hencky_2):
  simple_shear_test(hencky_1)
  simple_shear_test(hencky_2)


# def test_uniaxial_strain(gent_1, gent_2):
#   uniaxial_strain_test(gent_1)
#   uniaxial_strain_test(gent_2)

