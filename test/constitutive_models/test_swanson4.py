# from pancax import FixedProperties
# from pancax import Swanson4, Swanson4FixedBulkModulus
# from .utils import *
# import jax
# import jax.numpy as jnp
# import pytest


# @pytest.fixture
# def swanson():
#     return Swanson4(cutoff_strain=0.01)


# @pytest.fixture
# def swanson_fbm():
#     return Swanson4FixedBulkModulus(bulk_modulus=10.0, cutoff_strain=0.01)


# @pytest.fixture
# def properties():
#     return FixedProperties([10.0, 0.93074321, -0.07673672, 0.10448312, 1.71691036])


# @pytest.fixture
# def properties_fbm():
#     return FixedProperties([0.93074321, -0.07673672, 0.10448312, 1.71691036])



# def simple_shear_test(model, props):
#     gammas = jnp.linspace(0.05, 1., 100)
#     Fs = jax.vmap(simple_shear)(gammas)
#     B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
#     Js = jax.vmap(model.jacobian)(Fs)
#     I1_bars = jax.vmap(model.I1_bar)(Fs)
#     psis = jax.vmap(model.energy, in_axes=(0, None))(Fs, props())
#     sigmas = jax.vmap(model.cauchy_stress, in_axes=(0, None))(Fs, props())

#     K, A1, P1, C1, R1 = model.unpack_properties(props())

#     for (psi, sigma, I1_bar, J, B_bar) in zip(psis, sigmas, I1_bars, Js, B_bars):
#         psi_an = K * (J * jnp.log(J) - J + 1.) + \
#                  1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
#                  1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
#         dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
#                     0.5 * C1 * (I1_bar / 3. - 1.)**R1
#         B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
#         sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)

#         assert jnp.allclose(psi, psi_an, atol=1e-3)
#         for i in range(3):
#             for j in range(3):
#                 assert jnp.allclose(sigma[i, j], sigma_an[i, j], atol=1e-3)


# def uniaxial_strain_test(model, props):
#     lambdas = jnp.linspace(1.2, 4., 100)
#     Fs = jax.vmap(uniaxial_strain)(lambdas)
#     B_bars = jax.vmap(lambda F: jnp.power(jnp.linalg.det(F), -2. / 3.) * F @ F.T)(Fs)
#     Js = jax.vmap(model.jacobian)(Fs)
#     I1_bars = jax.vmap(model.I1_bar)(Fs)
#     psis = jax.vmap(model.energy, in_axes=(0, None))(Fs, props())
#     sigmas = jax.vmap(model.cauchy_stress, in_axes=(0, None))(Fs, props())

#     K, A1, P1, C1, R1 = model.unpack_properties(props())

#     for (psi, sigma, I1_bar, J, B_bar) in zip(psis, sigmas, I1_bars, Js, B_bars):
#         psi_an = K * (J * jnp.log(J) - J + 1.) + \
#                  1.5 * A1 / (P1 + 1.) * (I1_bar / 3. - 1.)**(P1 + 1.) + \
#                  1.5 * C1 / (R1 + 1.) * (I1_bar / 3. - 1.)**(R1 + 1.)
#         dUdI1_bar = 0.5 * A1 * (I1_bar / 3. - 1.)**P1 + \
#                     0.5 * C1 * (I1_bar / 3. - 1.)**R1
#         B_bar_dev = B_bar - (1. / 3.) * jnp.trace(B_bar) * jnp.eye(3)
#         sigma_an = (2. / J) * dUdI1_bar * B_bar_dev + K * jnp.log(J) * jnp.eye(3)

#         assert jnp.allclose(psi, psi_an, atol=1e-3)
#         for i in range(3):
#             for j in range(3):
#                 assert jnp.allclose(sigma[i, j], sigma_an[i, j], atol=1e-3)


# def unpack_props_test(model, props):
#     props = model.unpack_properties(props())
#     assert props[0] == 10.0
#     assert props[1] == 0.93074321
#     assert props[2] == -0.07673672
#     assert props[3] == 0.10448312
#     assert props[4] == 1.71691036


# def test_swanson_unpack_props(
#     swanson, properties,
#     swanson_fbm, properties_fbm
# ):
#     unpack_props_test(swanson, properties)
#     unpack_props_test(swanson_fbm, properties_fbm)


# def test_simple_shear(
#     swanson, properties,
#     swanson_fbm, properties_fbm   
# ):
#     simple_shear_test(swanson, properties)
#     simple_shear_test(swanson_fbm, properties_fbm)


# def test_uniaxial_strain(
#     swanson, properties,
#     swanson_fbm, properties_fbm   
# ):
#     uniaxial_strain_test(swanson, properties)
#     uniaxial_strain_test(swanson_fbm, properties_fbm)
