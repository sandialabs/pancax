from jax import hessian
from jax import jacfwd
from jax import vmap
from .math.tensor_math import tensor_2D_to_3D
import jax.numpy as jnp


# single point methods
# def field_value(network, x, t):e

def create_field_methods(domain, bc_func):
  # normalization up front
  # TODO currently hard-coded to min-max normalization
  # TODO make this selectable
  x_mins = jnp.min(domain.coords, axis=0)
  x_maxs = jnp.max(domain.coords, axis=0)

  # single point methods
  def field_value(network, x, t):
    x_normalized = (x - x_mins) / (x_maxs - x_mins)
    z = network(jnp.hstack((x_normalized, t)))
    u = bc_func(x, t, z)
    return u

  def field_gradient(network, x, t):
    return jacfwd(field_value, argnums=1)(network, x, t)

  def field_hessian(network, x, t):
    return hessian(field_value, argnums=1)(network, x, t)

  def field_time_derivative(network, x, t):
    return jacfwd(field_value, argnums=2)(network, x, t)

  return field_value, field_gradient, field_hessian, field_time_derivative
  # all point methods at single time step
  # def field_values(network, xs, t):
  #   return vmap(field_value, in_axes=(None, 0, None))(network, xs, t)
  
  # def field_gradients(network, xs, t):
  #   return vmap(field_gradient, in_axes=(None, 0, None))(network, xs, t)

  # def field_hessians(network, xs, t):
  #   return vmap(field_hessian, in_axes=(None, 0, None))(network, xs, t)

  # def field_time_derivatives(network, xs, t):
  #   return vmap(field_time_derivative, in_axes=(None, 0, None))(network, xs, t)

  # return field_values, field_gradients, field_hessians, field_time_derivatives


def deformation_gradients(grad_us, formulation):
  temp = vmap(lambda x: tensor_2D_to_3D(x) + jnp.eye(3))(grad_us)
  # temp = tensor_2D_to_3D(grad_us) + jnp.eye(3)
  Fs = vmap(lambda x: formulation(x), in_axes=(0,))(temp)
  return Fs


def invariants_old(grad_us, formulation):
  Fs         = deformation_gradients(grad_us, formulation)
  Cs         = vmap(lambda F: F.T @ F, in_axes=(0,))(Fs)
  Cs_squared = vmap(lambda C: C @ C, in_axes=(0,))(Cs)
  I1s        = vmap(lambda C: jnp.trace(C), in_axes=(0,))(Cs)
  I2s        = vmap(lambda I1, C_squared: 0.5 * (I1**2 - jnp.trace(C_squared)), in_axes=(0, 0))(I1s, Cs_squared)
  I3s        = vmap(lambda C: jnp.linalg.det(C), in_axes=(0,))(Cs)
  return I1s, I2s, I3s


def invariants(grad_u):
  F         = grad_u + jnp.eye(3)
  C         = F.T @ F
  C_squared = C @ C
  I1        = jnp.trace(C)
  I2        = 0.5 * (I1**2 - jnp.trace(C_squared))
  I3        = jnp.linalg.det(C)
  return I1, I2, I3


def plane_strain(F):
  return F


def incompressible_plane_stress(F):
  F_33 = 1.0 / (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
  return jnp.array(
    [
      [F[0, 0], F[0, 1], 0.0],
      [F[1, 0], F[1, 1], 0.0],
      [0.0    , 0.0    , F_33]
    ]
  )