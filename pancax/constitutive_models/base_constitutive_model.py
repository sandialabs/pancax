from abc import ABC, abstractmethod
from jaxtyping import Array, Float
from typing import List
import jax
import jax.numpy as jnp


class ConstitutiveModel(ABC):
  """
  Base class for consistutive models. 

  The interface to be defined by derived classes include
  the energy method and the unpack_properties method.

  :param n_properties: The number of properties
  :param property_names: The names of the properties
  """
  n_properties: int
  property_names: List[str]

  def cauchy_stress(self, F: Float[Array, "3 3"], props: Float[Array, "np"]):
    J = self.jacobian(F)
    P = self.pk1_stress(F, props)
    return (1. / J) * P @ F.T

  # def deformation_gradient(self, grad_u: Float[Array, "3 3"]):
  #   return grad_u + jnp.eye(3)

  @abstractmethod
  def energy(self, F: Float[Array, "3 3"], props: Float[Array, "np"]):
    """
    This method returns the algorithmic strain energy density.
    """
    pass

  def invariants(self, F: Float[Array, "3 3"]):
    I1 = self.I1(F)
    I2 = self.I2(F)
    I3 = self.jacobian(F)**2
    return jnp.array([I1, I2, I3])

  def jacobian(self, F: Float[Array, "3 3"]):
    r"""
    This simply calculate the jacobian but with guard rails
    to return nonsensical numbers if a non-positive jacobian
    is encountered during training.

    :param F: the deformation gradient

    .. math::
      J = det(\mathbf{F})
    
    """
    J = jnp.linalg.det(F)
    J = jax.lax.cond(
      J <= 0.0,
      lambda _: 1.0e3,
      lambda x: x,
      J
    )
    return J

  def I1(self, F: Float[Array, "3 3"]):
    r"""
    Calculates the first invariant

    :param F: the deformation gradient
    
    .. math::
      I_1 = tr\left(\mathbf{F}^T\mathbf{F}\right)
    """
    I1 = jnp.trace(F @ F.T)
    return I1

  def I1_bar(self, F: Float[Array, "3 3"]):
    r"""
    Calculates the first distortional invariant

    :param F: the deformation gradient
    
    .. math::
      \bar{I}_1 = J^{-2/3}tr\left(\mathbf{F}^T\mathbf{F}\right)
    """
    I1 = jnp.trace(F @ F.T)
    J = self.jacobian(F)
    return jnp.power(J, -2. / 3.) * I1

  def I2(self, F: Float[Array, "3 3"]):
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    return I2

  def I2_bar(self, F: Float[Array, "3 3"]):
    C = F.T @ F
    C2 = C @ C
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    J = self.jacobian(F)
    return jnp.power(J, -4. / 3.) * I2

  def pk1_stress(self, F: Float[Array, "np"], props: Float[Array, "np"]):
    return jax.grad(self.energy, argnums=0)(F, props)

  @abstractmethod
  def unpack_properties(self, props: Float[Array, "np"]):
    """
    This method unpacks properties from 'props' and returns
    them with potentially static properties bound to the model.
    """
    pass


class ConstitutiveModelFixedBulkModulus(ConstitutiveModel):
  n_properties: int
  property_names: List[str]
  bulk_modulus: float

  def __init__(self, bulk_modulus: float) -> None:
    self.bulk_modulus = bulk_modulus
