from .base_loss_function import PhysicsLossFunction
from jax import vmap
from pancax.physics import potential_energy, potential_energy_and_residual
from pancax.physics import incompressible_energy, incompressible_energy_and_residual
from pancax.physics import potential_energy_residual_and_reaction_force
from pancax.physics import quadrature_incompressibility_constraint
from pancax.physics import residual_mse
from typing import Optional
import equinox as eqx
import jax.numpy as jnp


class EnergyLoss(PhysicsLossFunction):
  r"""
  Energy loss function akin to the deep energy method.

  Calculates the following quantity

  .. math::
    \mathcal{L} = w\Pi\left[u\right] = w\int_\Omega\psi\left(\mathbf{F}\right)

  :param weight: weight for this loss function
  """
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    energies = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    )
    energy = jnp.sum(energies)
    loss = energy
    return self.weight * loss, dict(energy=energy)

  def load_step(self, params, domain, t):
    # field_network, props = params
    # us = domain.physics.field_values(field_network, t)
    # us = domain.physics.vmap_field_values(field_network, domain.coords, t)
    # props = props()
    # pi = potential_energy(domain, us, props)
    field, physics = params
    us = physics.vmap_field_values(field, domain.coords, t)
    pi = physics.potential_energy(params, domain.domain, t, us)
    return pi


class EnergyAndResidualLoss(PhysicsLossFunction):
  r"""
  Energy and residual loss function used in Hamel et. al

  Calculates the following quantity

  .. math::
    \mathcal{L} = w_1\Pi\left[u\right] + w_2\delta\Pi\left[u\right]_{free}

  :param energy_weight: Weight for the energy w_1
  :param residual_weight: Weight for the residual w_2
  """
  energy_weight: float
  residual_weight: float

  def __init__(
    self, 
    energy_weight: Optional[float] = 1.0,
    residual_weight: Optional[float] = 1.0
  ):
    self.energy_weight = energy_weight
    self.residual_weight = residual_weight

  def __call__(self, params, domain):
    pis, Rs = vmap(self.load_step, in_axes=(None, None, 0))(params, domain, domain.times)
    pi, R = jnp.sum(pis), jnp.sum(Rs)
    loss = self.energy_weight * pi + self.residual_weight * R
    return loss, dict(energy=pi, residual=R)

  def load_step(self, params, domain, t):
    field_network, props = params
    us = domain.field_values(field_network, t)
    props = props()
    pi, R = potential_energy_and_residual(domain, us, props)
    return pi, R
  

class EnergyResidualAndReactionLoss(PhysicsLossFunction):
  energy_weight: float
  residual_weight: float
  reaction_weight: float
  
  def __init__(
    self, 
    energy_weight: Optional[float] = 1.0,
    residual_weight: Optional[float] = 1.0,
    reaction_weight: Optional[float] = 1.0
  ):
    self.energy_weight = energy_weight
    self.residual_weight = residual_weight
    self.reaction_weight = reaction_weight

  def __call__(self, params, domain):
    pis, Rs, reactions = vmap(self.load_step, in_axes=(None, None, 0))(params, domain, domain.times)
    pi, R = jnp.sum(pis), jnp.sum(Rs) / len(domain.times)
    reaction_loss = jnp.square(reactions - domain.global_data.outputs).mean()
    loss = self.energy_weight * pi + \
           self.residual_weight * R + \
           self.reaction_weight * reaction_loss
    return loss, dict(energy=pi, residual=R, global_data_loss=reaction_loss, reactions=reactions)

  def load_step(self, params, domain, t):
    # field_network, props = params
    field, physics = params
    # us = domain.field_values(field_network, t)
    us = physics.vmap_field_values(field, domain.coords, t)
    return physics.potential_energy_residual_and_reaction_force(params, domain, t, us, domain.global_data)
    # props = props()
    # return potential_energy_residual_and_reaction_force(domain, us, props)


class QuadratureIncompressibilityConstraint(PhysicsLossFunction):
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    losses = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    )
    loss = jnp.mean(losses)
    return self.weight * loss, dict(incompressibility_error=loss)

  def load_step(self, params, domain, t):
    field_network, props = params
    us = domain.field_values(field_network, t)
    props = props()
    return quadrature_incompressibility_constraint(domain, us, props)


# class QuadratureIncompressibilityConstraintWithForce(PhysicsLossFunction):
#   energy_weight: float
#   residual_weight: float

#   def __init__(
#     self, 
#     energy_weight: Optional[float] = 1.0,
#     residual_weight: Optional[float] = 1.0
#   ):
#     self.energy_weight = energy_weight
#     self.residual_weight = residual_weight

#   def __call__(self, params, domain):
#     losses, residuals = vmap(self.load_step, in_axes=(None, None, 0))(
#       params, domain, domain.times
#     )
#     loss, R = jnp.sum(losses), jnp.sum(residuals)
#     # return self.weight * loss, dict(incompressibility_error=loss)
#     loss = self.energy_weight * loss + self.residual_weight * R
#     return loss, dict(constraint_energy=loss, constraint_force=R)

#   def load_step(self, params, domain, t):
#     field_network, props = params
#     us = domain.field_values(field_network, t)
#     props = props()
#     return quadrature_incompressibility_constraint_energy_and_residual(domain, us, props)

#   def __call__(self, params, domain):
#     pis, Rs = vmap(self.load_step, in_axes=(None, None, 0))(params, domain, domain.times)
#     pi, R = jnp.sum(pis), jnp.sum(Rs)
#     loss = self.energy_weight * pi + self.residual_weight * R
#     return loss, dict(energy=pi, residual=R)

#   def load_step(self, params, domain, t):
#     field_network, props = params
#     us = domain.field_values(field_network, t)
#     props = props()
#     pi, R = potential_energy_and_residual(domain, us, props)
#     return pi, R


class ResidualMSELoss(PhysicsLossFunction):
  weight: float

  def __init__(
    self, 
    weight: Optional[float] = 1.0
  ):
    self.weight = weight

  def __call__(self, params, domain):
    residuals = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    )
    residual = jnp.sum(residuals)
    loss = residual
    return self.weight * loss, dict(residual=residual)

  def load_step(self, params, domain, t):
    field_network, props = params
    us = domain.field_values(field_network, t)
    props = props()
    R = residual_mse(domain, us, props)
    return R


class IncompressibleEnergyLoss(PhysicsLossFunction):
  r"""
  Energy loss function akin to the deep energy method.

  Calculates the following quantity

  .. math::
    \mathcal{L} = w\Pi\left[u\right] = w\int_\Omega\psi\left(\mathbf{F}\right)

  :param weight: weight for this loss function
  """
  weight: float

  def __init__(self, weight: Optional[float] = 1.0):
    self.weight = weight

  def __call__(self, params, domain):
    energies = vmap(self.load_step, in_axes=(None, None, 0))(
      params, domain, domain.times
    )
    energy = jnp.sum(energies)
    loss = energy
    return self.weight * loss, dict(energy=energy)

  def load_step(self, params, domain, t):
    field_network, props = params
    us = domain.field_values(field_network, t)
    props = props()
    pi = incompressible_energy(domain, us, props)
    return pi


class IncompressibleEnergyAndResidualLoss(PhysicsLossFunction):
  r"""
  Energy and residual loss function used in Hamel et. al

  Calculates the following quantity

  .. math::
    \mathcal{L} = w_1\Pi\left[u\right] + w_2\delta\Pi\left[u\right]_{free}

  :param energy_weight: Weight for the energy w_1
  :param residual_weight: Weight for the residual w_2
  """
  energy_weight: float
  residual_weight: float

  def __init__(
    self, 
    energy_weight: Optional[float] = 1.0,
    residual_weight: Optional[float] = 1.0
  ):
    self.energy_weight = energy_weight
    self.residual_weight = residual_weight

  def __call__(self, params, domain):
    pis, Rs = vmap(self.load_step, in_axes=(None, None, 0))(params, domain, domain.times)
    pi, R = jnp.sum(pis), jnp.sum(Rs)
    loss = self.energy_weight * pi + self.residual_weight * R
    return loss, dict(energy=pi, residual=R)

  def load_step(self, params, domain, t):
    field_network, props = params
    us = domain.field_values(field_network, t)
    props = props()
    pi, R = incompressible_energy_and_residual(domain, us, props)
    return pi, R
  