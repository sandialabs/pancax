import pytest


@pytest.fixture
def R():
    from scipy.spatial.transform import Rotation
    return Rotation.random(random_state=41).as_matrix()


def numerical_grad(f):
    import jax.numpy as jnp

    def lam(A):
        df = jnp.zeros((3, 3))
        eps = 1e-7
        ff = f(A)
        for i in range(3):
            for j in range(3):
                Ap = A.at[i, j].add(eps)
                fp = f(Ap)
                fprime = (fp - ff) / eps
                df = df.at[i, j].add(fprime)
        return df

    return lam


def generate_n_random_symmetric_matrices(n, minval=0.0, maxval=1.0):
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    As = jax.random.uniform(key, (n, 3, 3), minval=minval, maxval=maxval)
    return jax.vmap(lambda A: jnp.dot(A.T, A), (0,))(As)


def log_squared(A):
    from pancax.math import tensor_math
    import jax.numpy as jnp
    return jnp.tensordot(tensor_math.log_sqrt(A), tensor_math.log_sqrt(A))


def test_log_sqrt_tensor_jvp_0():
    from jax.test_util import check_grads
    import jax.numpy as jnp
    A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 2.0]])
    check_grads(log_squared, (A,), order=1)


def test_log_sqrt_tensor_jvp_1():
    from jax.test_util import check_grads
    import jax.numpy as jnp
    A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 3.0]])
    check_grads(log_squared, (A,), order=1)


def test_log_sqrt_tensor_jvp_2():
    from jax.test_util import check_grads
    import jax.numpy as jnp
    A = jnp.array([[2.0, 0.0, 0.2], [0.0, 1.2, 0.1], [0.2, 0.1, 3.0]])
    check_grads(log_squared, (A,), order=1)


# @unittest.expectedFailure
# def test_log_sqrt_hessian_on_double_degenerate_eigenvalues(self):
#     eigvals = np.array([2., 0.5, 2.])
#     C = R@np.diag(eigvals)@R.T
#     check_grads(jax.jacrev(
# TensorMath.log_sqrt), (C,), order=1,
# modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


def test_eigen_sym33_non_unit():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    C = F.T @ F
    d, vecs = tensor_math.eigen_sym33_unit(C)
    jnp.array_equal(C, vecs @ jnp.diag(d) @ vecs.T)
    jnp.array_equal(vecs @ vecs.T, jnp.identity(3))


def test_eigen_sym33_non_unit_degenerate_case():
    from pancax.math import tensor_math
    import jax.numpy as jnp
    C = 5.0 * jnp.identity(3)
    d, vecs = tensor_math.eigen_sym33_unit(C)
    jnp.array_equal(C, vecs @ jnp.diag(d) @ vecs.T)
    jnp.array_equal(vecs @ vecs.T, jnp.identity(3))


# mtk_log_sqrt tests #


def test_log_sqrt_scaled_identity():
    from pancax.math import tensor_math
    import jax.numpy as jnp
    val = 1.2
    C = jnp.diag(jnp.array([val, val, val]))
    logSqrtVal = jnp.log(jnp.sqrt(val))
    jnp.array_equal(
        tensor_math.mtk_log_sqrt(C),
        jnp.diag(jnp.array([logSqrtVal, logSqrtVal, logSqrtVal])),
    )


def test_log_sqrt_double_eigs(R):
    from pancax.math import tensor_math
    import jax.numpy as jnp
    val1 = 2.0
    val2 = 0.5
    C = R @ jnp.diag(jnp.array([val1, val2, val1])) @ R.T

    logSqrt1 = jnp.log(jnp.sqrt(val1))
    logSqrt2 = jnp.log(jnp.sqrt(val2))
    diagLogSqrt = jnp.diag(jnp.array([logSqrt1, logSqrt2, logSqrt1]))

    logSqrtCExpected = R @ diagLogSqrt @ R.T
    jnp.array_equal(tensor_math.mtk_log_sqrt(C), logSqrtCExpected)


def test_log_sqrt_squared_grad_scaled_identity():
    from pancax.math import tensor_math
    from jax.test_util import check_grads
    import jax.numpy as jnp
    val = 1.2
    C = jnp.diag(jnp.array([val, val, val]))

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return jnp.tensordot(lg, lg)

    check_grads(log_squared, (C,), order=1)


def test_log_sqrt_squared_grad_double_eigs(R):
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    val1 = 2.0
    val2 = 0.5
    C = R @ jnp.diag(jnp.array([val1, val2, val1])) @ R.T

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return jnp.tensordot(lg, lg)

    check_grads(log_squared, (C,), order=1)


def test_log_sqrt_squared_grad_rand():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    C = F.T @ F

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return jnp.tensordot(lg, lg)

    check_grads(log_squared, (C,), order=1)


# mtk_pow tests #


def test_pow_scaled_identity():
    from pancax.math import tensor_math
    import jax.numpy as jnp
    m = 0.25
    val = 1.2
    C = jnp.diag(jnp.array([val, val, val]))

    powVal = jnp.power(val, m)
    jnp.array_equal(
        tensor_math.mtk_pow(C, m),
        jnp.diag(jnp.array([powVal, powVal, powVal]))
    )


def test_pow_double_eigs(R):
    from pancax.math import tensor_math
    import jax.numpy as jnp
    m = 0.25
    val1 = 2.1
    val2 = 0.6
    C = R @ jnp.diag(jnp.array([val1, val2, val1])) @ R.T

    powVal1 = jnp.power(val1, m)
    powVal2 = jnp.power(val2, m)
    diagLogSqrt = jnp.diag(jnp.array([powVal1, powVal2, powVal1]))

    logSqrtCExpected = R @ diagLogSqrt @ R.T

    jnp.array_equal(tensor_math.mtk_pow(C, m), logSqrtCExpected)


def test_pow_squared_grad_scaled_identity():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    val = 1.2
    C = jnp.diag(jnp.array([val, val, val]))

    def pow_squared(A):
        m = 0.25
        lg = tensor_math.mtk_pow(A, m)
        return jnp.tensordot(lg, lg)

    check_grads(pow_squared, (C,), order=1)


def test_pow_squared_grad_double_eigs(R):
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    val1 = 2.0
    val2 = 0.5
    C = R @ jnp.diag(jnp.array([val1, val2, val1])) @ R.T

    def pow_squared(A):
        m = 0.25
        lg = tensor_math.mtk_pow(A, m)
        return jnp.tensordot(lg, lg)

    check_grads(pow_squared, (C,), order=1)


def test_pow_squared_grad_rand():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    C = F.T @ F

    def pow_squared(A):
        m = 0.25
        lg = tensor_math.mtk_pow(A, m)
        return jnp.tensordot(lg, lg)

    check_grads(pow_squared, (C,), order=1)


# sqrtm #


def test_sqrtm_jit():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    C = generate_n_random_symmetric_matrices(1)[0]
    sqrtm_jit = jax.jit(tensor_math.sqrtm)
    sqrtC = sqrtm_jit(C)
    assert not jnp.isnan(sqrtC).any()


def test_sqrtm():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    mats = generate_n_random_symmetric_matrices(100)
    sqrtm_jit = jax.jit(tensor_math.sqrtm)
    sqrtMats = jax.vmap(sqrtm_jit, (0,))(mats)
    shouldBeMats = jax.vmap(lambda A: jnp.dot(A, A), (0,))(sqrtMats)
    jnp.array_equal(shouldBeMats, mats)


def test_sqrtm_fwd_mode_derivative():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["fwd"])


def test_sqrtm_rev_mode_derivative():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["rev"])


def test_sqrtm_on_degenerate_eigenvalues(R):
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    C = R @ jnp.diag(jnp.array([2.0, 0.5, 2])) @ R.T
    sqrtC = tensor_math.sqrtm(C)
    shouldBeC = jnp.dot(sqrtC, sqrtC)
    jnp.array_equal(shouldBeC, C)
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["rev"])


def test_sqrtm_on_10x10():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10, 10), minval=1e-8, maxval=10.0)
    C = F.T @ F
    sqrtC = tensor_math.sqrtm(C)
    shouldBeC = jnp.dot(sqrtC, sqrtC)
    jnp.array_equal(shouldBeC, C)


def test_sqrtm_derivatives_on_10x10():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10, 10), minval=1e-8, maxval=10.0)
    C = F.T @ F
    check_grads(tensor_math.sqrtm, (C,), order=1, modes=["fwd", "rev"])


def test_logm_iss_on_matrix_near_identity():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    id_perturbation = 1.0 + jax.random.uniform(
        key, (3,), minval=1e-8, maxval=0.01
    )
    A = jnp.diag(id_perturbation)
    logA = tensor_math.logm_iss(A)
    jnp.array_equal(logA, jnp.diag(jnp.log(id_perturbation)))


def test_logm_iss_on_double_degenerate_eigenvalues(R):
    from pancax.math import tensor_math
    import jax.numpy as jnp
    eigvals = jnp.array([2.0, 0.5, 2.0])
    C = R @ jnp.diag(eigvals) @ R.T
    logC = tensor_math.logm_iss(C)
    logCSpectral = R @ jnp.diag(jnp.log(eigvals)) @ R.T
    jnp.array_equal(logC, logCSpectral)


def test_logm_iss_on_triple_degenerate_eigvalues():
    from pancax.math import tensor_math
    import jax.numpy as jnp
    A = 4.0 * jnp.identity(3)
    logA = tensor_math.logm_iss(A)
    jnp.array_equal(logA, jnp.log(4.0) * jnp.identity(3))


def test_logm_iss_jit():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    C = generate_n_random_symmetric_matrices(1)[0]
    logm_iss_jit = jax.jit(tensor_math.logm_iss)
    logC = logm_iss_jit(C)
    assert not jnp.isnan(logC).any()


def test_logm_iss_on_full_3x3s():
    from jax.scipy import linalg
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    mats = generate_n_random_symmetric_matrices(1000)
    logm_iss_jit = jax.jit(tensor_math.logm_iss)
    logMats = jax.vmap(logm_iss_jit, (0,))(mats)
    shouldBeMats = jax.vmap(lambda A: linalg.expm(A), (0,))(logMats)
    # self.assertArrayNear(shouldBeMats, mats, 7)
    jnp.array_equal(shouldBeMats, mats)


def test_logm_iss_fwd_mode_derivative():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    C = generate_n_random_symmetric_matrices(1)[0]
    logm_iss_jit = jax.jit(tensor_math.logm_iss)
    check_grads(logm_iss_jit, (C,), order=1, modes=["fwd"])


def test_logm_iss_rev_mode_derivative():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    C = generate_n_random_symmetric_matrices(1)[0]
    logm_iss_jit = jax.jit(tensor_math.logm_iss)
    check_grads(logm_iss_jit, (C,), order=1, modes=["rev"])


def test_logm_iss_hessian_on_double_degenerate_eigenvalues(R):
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    eigvals = jnp.array([2.0, 0.5, 2.0])
    C = R @ jnp.diag(eigvals) @ R.T
    check_grads(
        jax.jacrev(tensor_math.logm_iss),
        (C,),
        order=1,
        modes=["fwd"],
        rtol=1e-9,
        atol=1e-9,
        eps=1e-5,
    )


def test_logm_iss_derivatives_on_double_degenerate_eigenvalues(R):
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    eigvals = jnp.array([2.0, 0.5, 2.0])
    C = R @ jnp.diag(eigvals) @ R.T
    check_grads(tensor_math.logm_iss, (C,), order=1, modes=["fwd"])
    check_grads(tensor_math.logm_iss, (C,), order=1, modes=["rev"])


def test_logm_iss_derivatives_on_triple_degenerate_eigenvalues():
    from jax.test_util import check_grads
    from pancax.math import tensor_math
    import jax.numpy as jnp
    A = 4.0 * jnp.identity(3)
    check_grads(tensor_math.logm_iss, (A,), order=1, modes=["fwd"])
    check_grads(tensor_math.logm_iss, (A,), order=1, modes=["rev"])


def test_logm_iss_on_10x10():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10, 10), minval=1e-8, maxval=10.0)
    C = F.T @ F
    logC = tensor_math.logm_iss(C)
    logCSpectral = tensor_math.logh(C)
    jnp.array_equal(logC, logCSpectral)


def test_compute_deviatoric_tensor():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert jnp.allclose(
        tensor_math.compute_deviatoric_tensor(F),
        F - (1.0 / 3.0) * jnp.trace(F) * jnp.eye(3),
    )


def test_tensor_norm():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert jnp.allclose(
        tensor_math.tensor_norm(F), jnp.linalg.norm(F, ord="fro")
    )


def test_norm_of_deviator_squared():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    dev = tensor_math.compute_deviatoric_tensor(F)
    an = jnp.tensordot(dev, dev)
    assert jnp.allclose(tensor_math.norm_of_deviator_squared(F), an)


def test_norm_of_deviator():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    an = tensor_math.tensor_norm(tensor_math.compute_deviatoric_tensor(F))
    assert jnp.allclose(tensor_math.norm_of_deviator(F), an)


def test_mises_equivalent_stress():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    an = jnp.sqrt(1.5) * tensor_math.norm_of_deviator(F)
    assert jnp.allclose(tensor_math.mises_equivalent_stress(F), an)


def test_triaxiality():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    mean_normal = jnp.trace(F) / 3.0
    an = mean_normal / (
        tensor_math.mises_equivalent_stress(F) +
        jnp.finfo(jnp.dtype("float64")).eps
    )
    assert jnp.allclose(tensor_math.triaxiality(F), an)


def test_sym():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert jnp.allclose(tensor_math.sym(F), 0.5 * (F + F.T))


def test_tensor_2D_to_3D():
    from pancax.math import tensor_math
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (2, 2), minval=1e-8, maxval=10.0)
    F_3D = tensor_math.tensor_2D_to_3D(F)
    assert jnp.allclose(F_3D[0, 0], F[0, 0])
    assert jnp.allclose(F_3D[0, 1], F[0, 1])
    assert jnp.allclose(F_3D[1, 0], F[1, 0])
    assert jnp.allclose(F_3D[1, 1], F[1, 1])
    #
    assert jnp.allclose(F_3D[0, 2], 0.0)
    assert jnp.allclose(F_3D[1, 2], 0.0)
    assert jnp.allclose(F_3D[2, 0], 0.0)
    assert jnp.allclose(F_3D[2, 1], 0.0)
    assert jnp.allclose(F_3D[2, 2], 0.0)
