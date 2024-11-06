
from jax import numpy as np
from jax.scipy import linalg
from jax.test_util import check_grads
from pancax.math import tensor_math
from scipy.spatial.transform import Rotation
import jax


R = Rotation.random(random_state=41).as_matrix()

def numerical_grad(f):
    def lam(A):
        df = np.zeros((3,3))
        eps = 1e-7
        ff = f(A)
        for i in range(3):
            for j in range(3):
                Ap = A.at[i,j].add(eps)
                fp = f(Ap)
                fprime = (fp-ff)/eps
                df = df.at[i,j].add(fprime)
        return df
    return lam


def generate_n_random_symmetric_matrices(n, minval=0.0, maxval=1.0):
    key = jax.random.PRNGKey(0)
    As = jax.random.uniform(key, (n,3,3), minval=minval, maxval=maxval)
    return jax.vmap(lambda A: np.dot(A.T,A), (0,))(As)


log_squared = lambda A: np.tensordot(tensor_math.log_sqrt(A), tensor_math.log_sqrt(A))
sqrtm_jit = jax.jit(tensor_math.sqrtm)
logm_iss_jit = jax.jit(tensor_math.logm_iss)


def test_log_sqrt_tensor_jvp_0():
    A = np.array([[2.0, 0.0, 0.0],
                  [0.0, 1.2, 0.0],
                  [0.0, 0.0, 2.0]])
    check_grads(log_squared, (A,), order=1)
    
    
def test_log_sqrt_tensor_jvp_1():
    A = np.array([[2.0, 0.0, 0.0],
                  [0.0, 1.2, 0.0],
                  [0.0, 0.0, 3.0]])
    check_grads(log_squared, (A,), order=1)

    
def test_log_sqrt_tensor_jvp_2():
    A = np.array([[2.0, 0.0, 0.2],
                  [0.0, 1.2, 0.1],
                  [0.2, 0.1, 3.0]])
    check_grads(log_squared, (A,), order=1)


# @unittest.expectedFailure
# def test_log_sqrt_hessian_on_double_degenerate_eigenvalues(self):
#     eigvals = np.array([2., 0.5, 2.])
#     C = R@np.diag(eigvals)@R.T
#     check_grads(jax.jacrev(TensorMath.log_sqrt), (C,), order=1, modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


def test_eigen_sym33_non_unit():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
    C = F.T@F
    d,vecs = tensor_math.eigen_sym33_unit(C)
    np.array_equal(C, vecs @ np.diag(d) @ vecs.T)
    np.array_equal(vecs @ vecs.T, np.identity(3))
    

def test_eigen_sym33_non_unit_degenerate_case():
    C = 5.0*np.identity(3)
    d,vecs = tensor_math.eigen_sym33_unit(C)
    np.array_equal(C, vecs @ np.diag(d) @ vecs.T)
    np.array_equal(vecs @ vecs.T, np.identity(3))
    
### mtk_log_sqrt tests ###
    
    
def test_log_sqrt_scaled_identity():
    val = 1.2
    C = np.diag(np.array([val, val, val]))
    logSqrtVal = np.log(np.sqrt(val))
    np.array_equal(tensor_math.mtk_log_sqrt(C), np.diag(np.array([logSqrtVal, logSqrtVal, logSqrtVal])))


def test_log_sqrt_double_eigs():
    val1 = 2.0
    val2 = 0.5
    C = R@np.diag(np.array([val1, val2, val1]))@R.T

    logSqrt1 = np.log(np.sqrt(val1))
    logSqrt2 = np.log(np.sqrt(val2))
    diagLogSqrt = np.diag(np.array([logSqrt1, logSqrt2, logSqrt1]))

    logSqrtCExpected = R@diagLogSqrt@R.T
    np.array_equal(tensor_math.mtk_log_sqrt(C), logSqrtCExpected)

    
def test_log_sqrt_squared_grad_scaled_identity():
    val = 1.2
    C = np.diag(np.array([val, val, val]))

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return np.tensordot(lg, lg)
    check_grads(log_squared, (C,), order=1)
    
    
def test_log_sqrt_squared_grad_double_eigs():
    val1 = 2.0
    val2 = 0.5
    C = R@np.diag(np.array([val1, val2, val1]))@R.T

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return np.tensordot(lg, lg)
    check_grads(log_squared, (C,), order=1)

    
def test_log_sqrt_squared_grad_rand():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
    C = F.T@F

    def log_squared(A):
        lg = tensor_math.mtk_log_sqrt(A)
        return np.tensordot(lg, lg)
    check_grads(log_squared, (C,), order=1)
    
    
### mtk_pow tests ###


def test_pow_scaled_identity():
    m = 0.25
    val = 1.2
    C = np.diag(np.array([val, val, val]))

    powVal = np.power(val, m)
    np.array_equal(tensor_math.mtk_pow(C,m), np.diag(np.array([powVal, powVal, powVal])))


def test_pow_double_eigs():
    m = 0.25
    val1 = 2.1
    val2 = 0.6
    C = R@np.diag(np.array([val1, val2, val1]))@R.T

    powVal1 = np.power(val1, m)
    powVal2 = np.power(val2, m)
    diagLogSqrt = np.diag(np.array([powVal1, powVal2, powVal1]))

    logSqrtCExpected = R@diagLogSqrt@R.T

    np.array_equal(tensor_math.mtk_pow(C,m), logSqrtCExpected)

    
def test_pow_squared_grad_scaled_identity():
    val = 1.2
    C = np.diag(np.array([val, val, val]))

    def pow_squared(A):
        m = 0.25
        lg = tensor_math.mtk_pow(A, m)
        return np.tensordot(lg, lg)
    check_grads(pow_squared, (C,), order=1)


def test_pow_squared_grad_double_eigs():
    val1 = 2.0
    val2 = 0.5
    C = R@np.diag(np.array([val1, val2, val1]))@R.T

    def pow_squared(A):
        m=0.25
        lg = tensor_math.mtk_pow(A, m)
        return np.tensordot(lg, lg)
    check_grads(pow_squared, (C,), order=1)


def test_pow_squared_grad_rand():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
    C = F.T@F

    def pow_squared(A):
        m=0.25
        lg = tensor_math.mtk_pow(A, m)
        return np.tensordot(lg, lg)
    check_grads(pow_squared, (C,), order=1)
    
    
### sqrtm ###


def test_sqrtm_jit():
    C = generate_n_random_symmetric_matrices(1)[0]
    sqrtC = sqrtm_jit(C)
    assert not np.isnan(sqrtC).any()
    

def test_sqrtm():
    mats = generate_n_random_symmetric_matrices(100)
    sqrtMats = jax.vmap(sqrtm_jit, (0,))(mats)
    shouldBeMats = jax.vmap(lambda A: np.dot(A, A), (0,))(sqrtMats)
    np.array_equal(shouldBeMats, mats)


def test_sqrtm_fwd_mode_derivative():
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["fwd"])


def test_sqrtm_rev_mode_derivative():
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["rev"])


def test_sqrtm_on_degenerate_eigenvalues():
    C = R@np.diag(np.array([2., 0.5, 2]))@R.T
    sqrtC = tensor_math.sqrtm(C)
    shouldBeC = np.dot(sqrtC, sqrtC)
    np.array_equal(shouldBeC, C)
    check_grads(tensor_math.sqrtm, (C,), order=2, modes=["rev"])


def test_sqrtm_on_10x10():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
    C = F.T@F
    sqrtC = tensor_math.sqrtm(C)
    shouldBeC = np.dot(sqrtC,sqrtC)
    np.array_equal(shouldBeC, C)


def test_sqrtm_derivatives_on_10x10():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
    C = F.T@F
    check_grads(tensor_math.sqrtm, (C,), order=1, modes=["fwd", "rev"])


def test_logm_iss_on_matrix_near_identity():
    key = jax.random.PRNGKey(0)
    id_perturbation = 1.0 + jax.random.uniform(key, (3,), minval=1e-8, maxval=0.01)
    A = np.diag(id_perturbation)
    logA = tensor_math.logm_iss(A)
    np.array_equal(logA, np.diag(np.log(id_perturbation)))


def test_logm_iss_on_double_degenerate_eigenvalues():
    eigvals = np.array([2., 0.5, 2.])
    C = R@np.diag(eigvals)@R.T
    logC = tensor_math.logm_iss(C)
    logCSpectral = R@np.diag(np.log(eigvals))@R.T
    np.array_equal(logC, logCSpectral)


def test_logm_iss_on_triple_degenerate_eigvalues():
    A = 4.0*np.identity(3)
    logA = tensor_math.logm_iss(A)
    np.array_equal(logA, np.log(4.0)*np.identity(3))


def test_logm_iss_jit():
    C = generate_n_random_symmetric_matrices(1)[0]
    logC = logm_iss_jit(C)
    assert not np.isnan(logC).any()


def test_logm_iss_on_full_3x3s():
    mats = generate_n_random_symmetric_matrices(1000)
    logMats = jax.vmap(logm_iss_jit, (0,))(mats)
    shouldBeMats = jax.vmap(lambda A: linalg.expm(A), (0,))(logMats)
    # self.assertArrayNear(shouldBeMats, mats, 7)  
    np.array_equal(shouldBeMats, mats)    

    
def test_logm_iss_fwd_mode_derivative():
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(logm_iss_jit, (C,), order=1, modes=['fwd'])


def test_logm_iss_rev_mode_derivative():
    C = generate_n_random_symmetric_matrices(1)[0]
    check_grads(logm_iss_jit, (C,), order=1, modes=['rev'])


def test_logm_iss_hessian_on_double_degenerate_eigenvalues():
    eigvals = np.array([2., 0.5, 2.])
    C = R@np.diag(eigvals)@R.T
    check_grads(jax.jacrev(tensor_math.logm_iss), (C,), order=1, modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


def test_logm_iss_derivatives_on_double_degenerate_eigenvalues():
    eigvals = np.array([2., 0.5, 2.])
    C = R@np.diag(eigvals)@R.T
    check_grads(tensor_math.logm_iss, (C,), order=1, modes=['fwd'])
    check_grads(tensor_math.logm_iss, (C,), order=1, modes=['rev'])


def test_logm_iss_derivatives_on_triple_degenerate_eigenvalues():
    A = 4.0*np.identity(3)
    check_grads(tensor_math.logm_iss, (A,), order=1, modes=['fwd'])
    check_grads(tensor_math.logm_iss, (A,), order=1, modes=['rev'])


def test_logm_iss_on_10x10():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
    C = F.T@F
    logC = tensor_math.logm_iss(C)
    logCSpectral = tensor_math.logh(C)
    np.array_equal(logC, logCSpectral)


def test_compute_deviatoric_tensor():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert np.allclose(tensor_math.compute_deviatoric_tensor(F), F - (1. / 3.) * np.trace(F) * np.eye(3))


def test_tensor_norm():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert np.allclose(tensor_math.tensor_norm(F), np.linalg.norm(F, ord='fro'))


def test_norm_of_deviator_squared():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    dev = tensor_math.compute_deviatoric_tensor(F)
    an = np.tensordot(dev, dev)
    assert np.allclose(tensor_math.norm_of_deviator_squared(F), an)


def test_norm_of_deviator():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    an = tensor_math.tensor_norm(tensor_math.compute_deviatoric_tensor(F))
    assert np.allclose(tensor_math.norm_of_deviator(F), an)


def test_mises_equivalent_stress():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    an = np.sqrt(1.5) * tensor_math.norm_of_deviator(F)
    assert np.allclose(tensor_math.mises_equivalent_stress(F), an)


def test_triaxiality():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    mean_normal = np.trace(F) / 3.
    an = mean_normal / (tensor_math.mises_equivalent_stress(F) + np.finfo(np.dtype('float64')).eps)
    assert np.allclose(tensor_math.triaxiality(F), an)


def test_sym():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (3, 3), minval=1e-8, maxval=10.0)
    assert np.allclose(tensor_math.sym(F), 0.5 * (F + F.T))


def test_tensor_2D_to_3D():
    key = jax.random.PRNGKey(0)
    F = jax.random.uniform(key, (2, 2), minval=1e-8, maxval=10.0)
    F_3D = tensor_math.tensor_2D_to_3D(F)
    assert np.allclose(F_3D[0, 0], F[0, 0])
    assert np.allclose(F_3D[0, 1], F[0, 1])
    assert np.allclose(F_3D[1, 0], F[1, 0])
    assert np.allclose(F_3D[1, 1], F[1, 1])
    #
    assert np.allclose(F_3D[0, 2], 0.0)
    assert np.allclose(F_3D[1, 2], 0.0)
    assert np.allclose(F_3D[2, 0], 0.0)
    assert np.allclose(F_3D[2, 1], 0.0)
    assert np.allclose(F_3D[2, 2], 0.0)
