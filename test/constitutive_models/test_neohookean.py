from pancax import FixedProperties, NeoHookean, NeoHookeanFixedBulkModulus
from .utils import *
import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def neohookean():
    return NeoHookean()


@pytest.fixture
def neohookean_fbm():
    return NeoHookeanFixedBulkModulus(bulk_modulus=0.833)


@pytest.fixture
def properties():
    return FixedProperties([0.833, 0.3846])


@pytest.fixture
def properties_fbm():
    return FixedProperties([0.3846])


def simple_shear_test(model, props):
    gammas = jnp.linspace(0.0, 1., 100)
    Fs = jax.vmap(simple_shear)(gammas)
    Js = jax.vmap(model.jacobian)(Fs)
    I1_bars = jax.vmap(model.I1_bar)(Fs)
    psis = jax.vmap(model.energy, in_axes=(0, None))(Fs, props())
    sigmas = jax.vmap(model.cauchy_stress, in_axes=(0, None))(Fs, props())
    K, G = model.unpack_properties(props())

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


def uniaxial_strain_test(model, props):
    lambdas = jnp.linspace(1., 4., 100)
    Fs = jax.vmap(uniaxial_strain)(lambdas)
    Js = jax.vmap(model.jacobian)(Fs)
    I1_bars = jax.vmap(model.I1_bar)(Fs)
    psis = jax.vmap(model.energy, in_axes=(0, None))(Fs, props())
    sigmas = jax.vmap(model.cauchy_stress, in_axes=(0, None))(Fs, props())
    K, G = model.unpack_properties(props())

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


def unpack_props_test(model, props):
    props = model.unpack_properties(props())
    assert props[0] == 0.833
    assert props[1] == 0.3846


def test_neohookean_unpack_props(
    neohookean, properties,
    neohookean_fbm, properties_fbm
):
    unpack_props_test(neohookean, properties)
    unpack_props_test(neohookean_fbm, properties_fbm)


def test_simple_shear(
    neohookean, properties,
    neohookean_fbm, properties_fbm   
):
    simple_shear_test(neohookean, properties)
    simple_shear_test(neohookean_fbm, properties_fbm)


def test_uniaxial_strain(
    neohookean, properties,
    neohookean_fbm, properties_fbm   
):
    uniaxial_strain_test(neohookean, properties)
    uniaxial_strain_test(neohookean_fbm, properties_fbm)
