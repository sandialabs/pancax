from abc import abstractmethod
from ..base import ConstitutiveModel, Scalar, Tensor
from typing import Tuple
import jax
import jax.numpy as jnp


class MechanicsModel(ConstitutiveModel):
  def cauchy_stress(self, grad_u: Tensor, *args) -> Tensor:
    F = self.deformation_gradient(grad_u)
    J = self.jacobian(grad_u)
    P = self.pk1_stress(grad_u, *args)
    return (1. / J) * P @ F.T

  def deformation_gradient(self, grad_u: Tensor) -> Tensor:
    F = grad_u + jnp.eye(3)
    return F

  @abstractmethod
  def energy(self, grad_u: Tensor, *args) -> Scalar:
    """
    This method returns the algorithmic strain energy density.
    """
    pass

  def invariants(self, grad_u: Tensor) -> Tuple[Scalar, Scalar, Scalar]:
    I1 = self.I1(grad_u)
    I2 = self.I2(grad_u)
    I3 = self.I3(grad_u)
    return jnp.array([I1, I2, I3])

  def I1(self, grad_u: Tensor) -> Scalar:
    r"""
    Calculates the first invariant

    - **grad_u**: the displacement gradient
    
    $$
    I_1 = tr\left(\mathbf{F}^T\mathbf{F}\right)
    $$
    """
    F = self.deformation_gradient(grad_u)
    I1 = jnp.trace(F @ F.T)
    return I1

  def I1_bar(self, grad_u: Tensor) -> Scalar:
    r"""
    Calculates the first distortional invariant

    - **grad_u**: the displacement gradient

    $$
    \bar{I}_1 = J^{-2/3}tr\left(\mathbf{F}^T\mathbf{F}\right)
    $$
    """
    F = self.deformation_gradient(grad_u)
    I1 = jnp.trace(F @ F.T)
    J = self.jacobian(grad_u)
    return jnp.power(J, -2. / 3.) * I1

  def I2(self, grad_u: Tensor) -> Scalar:
    F = self.deformation_gradient(grad_u)
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    return I2

  def I2_bar(self, grad_u: Tensor) -> Scalar:
    F = self.deformation_gradient(grad_u)
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    J = self.jacobian(grad_u)
    return jnp.power(J, -4. / 3.) * I2

  def I3(self, grad_u: Tensor) -> Scalar:
    J = self.jacobian(grad_u)
    return J * J

  def jacobian(self, grad_u: Tensor) -> Scalar:
    r"""
    This simply calculate the jacobian but with guard rails
    to return nonsensical numbers if a non-positive jacobian
    is encountered during training.

    - **grad_u**: the displacement gradient

    $$
    J = det(\mathbf{F})
    $$
    """
    F = self.deformation_gradient(grad_u)
    J = jnp.linalg.det(F)
    J = jax.lax.cond(
      J <= 0.0,
      lambda _: 1.0e3,
      lambda x: x,
      J
    )
    return J

  def pk1_stress(self, grad_u: Tensor, *args) -> Tensor:
    return jax.grad(self.energy, argnums=0)(grad_u, *args)


class HyperelasticModel(MechanicsModel):
  pass
