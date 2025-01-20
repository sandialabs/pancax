from abc import abstractmethod
from jaxtyping import Array, Float
from typing import Tuple
import equinox as eqx
import jax
import jax.numpy as jnp


Scalar = float
Tensor = Float[Array, "3 3"]


class BaseConstitutiveModel(eqx.Module):
  def cauchy_stress(self, F: Tensor) -> Tensor:
    # F = grad_u + jnp.eye(3)
    J = self.jacobian(F)
    P = self.pk1_stress(F)
    return (1. / J) * P @ F.T

  @abstractmethod
  def energy(self, F: Tensor) -> Scalar:
    """
    This method returns the algorithmic strain energy density.
    """
    pass

  def invariants(self, F: Tensor) -> Tuple[Scalar, Scalar, Scalar]:
    I1 = self.I1(F)
    I2 = self.I2(F)
    I3 = self.jacobian(F)**2
    return jnp.array([I1, I2, I3])

  def I1(self, F: Tensor) -> Scalar:
    r"""
    Calculates the first invariant

    - **F**: the deformation gradient
    
    $$
    I_1 = tr\left(\mathbf{F}^T\mathbf{F}\right)
    $$
    """
    I1 = jnp.trace(F @ F.T)
    return I1

  def I1_bar(self, F: Tensor) -> Scalar:
    r"""
    Calculates the first distortional invariant

    - **F**: the deformation gradient

    $$
    \bar{I}_1 = J^{-2/3}tr\left(\mathbf{F}^T\mathbf{F}\right)
    $$
    """
    I1 = jnp.trace(F @ F.T)
    J = self.jacobian(F)
    return jnp.power(J, -2. / 3.) * I1

  def I2(self, F: Tensor) -> Scalar:
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    return I2

  def I2_bar(self, F: Tensor) -> Scalar:
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    J = self.jacobian(F)
    return jnp.power(J, -4. / 3.) * I2

  def jacobian(self, F: Tensor) -> Scalar:
    r"""
    This simply calculate the jacobian but with guard rails
    to return nonsensical numbers if a non-positive jacobian
    is encountered during training.

    - **F**: the deformation gradient

    $$
    J = det(\mathbf{F})
    $$
    """
    J = jnp.linalg.det(F)
    J = jax.lax.cond(
      J <= 0.0,
      lambda _: 1.0e3,
      lambda x: x,
      J
    )
    return J

  def pk1_stress(self, F: Tensor) -> Tensor:
    return jax.grad(self.energy, argnums=0)(F)

  def properties(self):
    return self.__dataclass_fields__
