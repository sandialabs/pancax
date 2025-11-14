import pytest


@pytest.fixture
def sm_helper():
    from pancax.constitutive_models import NeoHookean
    import jax.numpy as jnp
    model = NeoHookean(
        bulk_modulus=1000.,
        shear_modulus=1.
    )
    theta = 60.
    state_old = jnp.zeros(0)
    dt = 0.
    return model, theta, state_old, dt


def test_plane_strain_formulation_extract_stress(sm_helper):
    from pancax.physics_kernels import PlaneStrain
    import jax.numpy as jnp

    model, theta, state_old, dt = sm_helper
    formulation = PlaneStrain()

    grad_u = jnp.array([
        [1., -0.5],
        [0.25, -0.5]
    ])
    grad_u = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )
    P, state_new = model.pk1_stress(grad_u, theta, state_old, dt)
    P_ex = formulation.extract_stress(P)
    assert jnp.allclose(P[0:2, 0:2], P_ex, rtol=1e-13)


def test_plane_strain_formulation_modify_field_gradient(sm_helper):
    from pancax.physics_kernels import PlaneStrain
    from pancax.math.tensor_math import tensor_2D_to_3D
    import jax.numpy as jnp
    import jax.random as jr

    model, theta, state_old, dt = sm_helper
    formulation = PlaneStrain()

    key = jr.PRNGKey(0)
    grad_u = jr.uniform(key=key, shape=(2, 2))

    grad_u_check = tensor_2D_to_3D(grad_u)
    grad_u_test = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )
    assert jnp.allclose(grad_u_check, grad_u_test, rtol=1e-13)


def test_plane_stress_formulation_extract_stress(sm_helper):
    from pancax.physics_kernels import PlaneStress
    import jax.numpy as jnp

    model, theta, state_old, dt = sm_helper
    formulation = PlaneStress()

    grad_u = jnp.array([
        [1., -0.5],
        [0.25, -0.5]
    ])
    grad_u = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )
    P, state_new = model.pk1_stress(grad_u, theta, state_old, dt)
    P_ex = formulation.extract_stress(P)
    assert jnp.allclose(P[0:2, 0:2], P_ex, rtol=1e-13)


def test_plane_stress_formulation_modify_field_gradient(sm_helper):
    from pancax.physics_kernels import PlaneStress
    import jax.numpy as jnp
    import jax.random as jr

    model, theta, state_old, dt = sm_helper
    formulation = PlaneStress()

    key = jr.PRNGKey(0)
    grad_u = jr.uniform(key=key, shape=(2, 2))

    grad_u_test = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )

    assert jnp.allclose(grad_u_test[0, 2], 0., rtol=1e-13)
    assert jnp.allclose(grad_u_test[1, 2], 0., rtol=1e-13)
    assert jnp.allclose(grad_u_test[2, 0], 0., rtol=1e-13)
    assert jnp.allclose(grad_u_test[2, 1], 0., rtol=1e-13)

    F = model.deformation_gradient(grad_u_test)

    assert jnp.allclose(F[0, 0], grad_u[0, 0] + 1., rtol=1e-13)
    assert jnp.allclose(F[0, 1], grad_u[0, 1], rtol=1e-13)
    assert jnp.allclose(F[1, 0], grad_u[1, 0], rtol=1e-13)
    assert jnp.allclose(F[1, 1], grad_u[1, 1] + 1., rtol=1e-13)

    assert jnp.allclose(F[0, 2], 0., rtol=1e-13)
    assert jnp.allclose(F[1, 2], 0., rtol=1e-13)
    assert jnp.allclose(F[2, 0], 0., rtol=1e-13)
    assert jnp.allclose(F[2, 1], 0., rtol=1e-13)

    assert jnp.allclose(
        1. / (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]), F[2, 2],
        rtol=1e-2
    )


def test_three_dimensional_formulation_extract_stress(sm_helper):
    from pancax.physics_kernels import ThreeDimensional
    import jax.numpy as jnp

    model, theta, state_old, dt = sm_helper
    formulation = ThreeDimensional()

    grad_u = jnp.array([
        [1., -0.5, 0.],
        [0.25, -0.5, 0.],
        [0., 0., 0.4]
    ])
    grad_u = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )
    P, state_new = model.pk1_stress(grad_u, theta, state_old, dt)
    P_ex = formulation.extract_stress(P)
    assert jnp.allclose(P, P_ex, rtol=1e-13)


def test_three_dimensional_formulation_modify_field_gradient(sm_helper):
    from pancax.physics_kernels import ThreeDimensional
    import jax.numpy as jnp
    import jax.random as jr

    model, theta, state_old, dt = sm_helper
    formulation = ThreeDimensional()

    key = jr.PRNGKey(0)
    grad_u = jr.uniform(key=key, shape=(3, 3))

    grad_u_check = grad_u
    grad_u_test = formulation.modify_field_gradient(
        model, grad_u, theta, state_old, dt
    )
    assert jnp.allclose(grad_u_check, grad_u_test, rtol=1e-13)
