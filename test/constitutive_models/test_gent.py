import pytest


K = 0.833
G = 0.3846
Jm = 3.0


@pytest.fixture
def gent_1():
    from pancax import Gent

    return Gent(bulk_modulus=K, shear_modulus=G, Jm_parameter=Jm)


@pytest.fixture
def gent_2():
    from pancax import BoundedProperty, Gent
    import jax

    key = jax.random.PRNGKey(0)
    return Gent(
        bulk_modulus=BoundedProperty(K, K, key),
        shear_modulus=BoundedProperty(G, G, key),
        Jm_parameter=BoundedProperty(Jm, Jm, key),
    )


def simple_shear_test(model):
    from .utils import simple_shear
    import jax
    import jax.numpy as jnp

    theta = 0.0
    state_old = jnp.zeros((100, 0))
    dt = 1.0
    gammas = jnp.linspace(0.0, 1.0, 100)
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

    def vmap_func(gamma, I1_bar, J):
        psi_an = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J)) + \
            -0.5 * G * Jm * jnp.log(
            1.0 - (I1_bar - 3.0) / Jm
        )
        sigma_11_an = 2.0 / 3.0 * G * Jm * gamma**2 / (Jm - gamma**2)
        sigma_22_an = -1.0 / 3.0 * G * Jm * gamma**2 / (Jm - gamma**2)
        sigma_12_an = G * Jm * gamma / (Jm - gamma**2)
        return psi_an, sigma_11_an, sigma_22_an, sigma_12_an

    psi_ans, sigma_11_ans, sigma_22_ans, sigma_12_ans = jax.vmap(
        vmap_func, in_axes=(0, 0, 0)
    )(gammas, I1_bars, Js)

    assert jnp.allclose(psis, psi_ans)
    assert jnp.allclose(sigmas[:, 0, 0], sigma_11_ans)
    assert jnp.allclose(sigmas[:, 1, 1], sigma_22_ans)
    assert jnp.allclose(sigmas[:, 2, 2], sigma_22_ans)
    #
    assert jnp.allclose(sigmas[:, 0, 1], sigma_12_ans)
    assert jnp.allclose(sigmas[:, 1, 2], 0.0)
    assert jnp.allclose(sigmas[:, 2, 0], 0.0)
    # #
    assert jnp.allclose(sigmas[:, 1, 0], sigma_12_ans)
    assert jnp.allclose(sigmas[:, 2, 1], 0.0)
    assert jnp.allclose(sigmas[:, 0, 2], 0.0)


def uniaxial_strain_test(model):
    from .utils import uniaxial_strain
    import jax
    import jax.numpy as jnp

    theta = 0.0
    state_old = jnp.zeros((100, 0))
    dt = 1.0
    lambdas = jnp.linspace(1.0, 2.0, 100)
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

    def vmap_func(lambda_, I1_bar, J):
        psi_an = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J)) + \
            -0.5 * G * Jm * jnp.log(
            1.0 - (I1_bar - 3.0) / Jm
        )
        sigma_11_an = 0.5 * K * (lambda_ - 1.0 / lambda_) - \
            2.0 / 3.0 * G * Jm * (
            lambda_**2 - 1.0
        ) / (lambda_**3 - (Jm + 3) * lambda_ ** (5.0 / 3.0) + 2.0 * lambda_)
        sigma_22_an = 0.5 * K * (lambda_ - 1.0 / lambda_) + \
            1.0 / 3.0 * G * Jm * (
            lambda_**2 - 1.0
        ) / (lambda_**3 - (Jm + 3) * lambda_ ** (5.0 / 3.0) + 2.0 * lambda_)
        return psi_an, sigma_11_an, sigma_22_an

    psi_ans, sigma_11_ans, sigma_22_ans = jax.vmap(
        vmap_func, in_axes=(0, 0, 0))(
        lambdas, I1_bars, Js
    )

    assert jnp.allclose(psis, psi_ans)
    assert jnp.allclose(sigmas[:, 0, 0], sigma_11_ans)
    assert jnp.allclose(sigmas[:, 1, 1], sigma_22_ans)
    assert jnp.allclose(sigmas[:, 2, 2], sigma_22_ans)
    #
    assert jnp.allclose(sigmas[:, 0, 1], 0.0)
    assert jnp.allclose(sigmas[:, 1, 2], 0.0)
    assert jnp.allclose(sigmas[:, 2, 0], 0.0)
    #
    assert jnp.allclose(sigmas[:, 1, 0], 0.0)
    assert jnp.allclose(sigmas[:, 2, 1], 0.0)
    assert jnp.allclose(sigmas[:, 0, 2], 0.0)


def test_simple_shear(gent_1, gent_2):
    simple_shear_test(gent_1)
    simple_shear_test(gent_2)


def test_uniaxial_strain(gent_1, gent_2):
    uniaxial_strain_test(gent_1)
    uniaxial_strain_test(gent_2)
