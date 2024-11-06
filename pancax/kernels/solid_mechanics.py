from abc import ABC, abstractmethod
from ..constitutive_models import ConstitutiveModel
from .base_kernel import StrongFormPhysicsKernel, WeakFormPhysicsKernel
from .base_kernel import element_pp, full_tensor_names, nodal_pp
from pancax.fem.function_space import compute_field_gradient
from pancax.math.tensor_math import *
from typing import Callable, List, Optional
import jax
import jax.numpy as jnp


# different formulations e.g. plane strain/stress, axisymmetric etc.
class BaseMechanicsFormulation(ABC):
	n_dimensions: int

	@abstractmethod
	def modify_field_gradient(self, grad_u):
		pass


# note for this formulation we're getting NaNs if the 
# reference configuration is used during calculation 
# of the loss function
class IncompressiblePlaneStress(BaseMechanicsFormulation):
	n_dimensions = 2

	def __init__(self) -> None:
		print('WARNING: Do not include a time of 0.0 with this formulation. You will get NaNs.')

	def deformation_gradient(self, grad_u):
		F = tensor_2D_to_3D(grad_u) + jnp.eye(3)
		F = F.at[2, 2].set(1. / jnp.linalg.det(grad_u + jnp.eye(2)))
		return F

	def modify_field_gradient(self, grad_u):
		F = self.deformation_gradient(grad_u)
		return F - jnp.eye(3)


class PlaneStrain(BaseMechanicsFormulation):
	n_dimensions = 2

	def extract_stress(self, P):
		return P[0:2, 0:2]

	def modify_field_gradient(self, grad_u):
		return tensor_2D_to_3D(grad_u)


class ThreeDimensional(BaseMechanicsFormulation):
	n_dimensions = 3

	def modify_field_gradient(self, grad_u):
		return grad_u


# actual mechanics kernel
class SolidMechanics(StrongFormPhysicsKernel, WeakFormPhysicsKernel):
	n_dofs: int
	field_values_names: List[str]
	bc_func: Callable
	constitutive_model: ConstitutiveModel
	formulation: BaseMechanicsFormulation
	var_name_to_method: dict
	use_delta_pinn: bool

	def __init__(
		self, 
		mesh, 
		bc_func, 
		constitutive_model, 
		formulation,
		use_delta_pinn: Optional[bool] = False
	):
		super().__init__(mesh, bc_func, use_delta_pinn)
		self.n_dofs = formulation.n_dimensions
		self.field_value_names = [
			'displ_x', 'displ_y'
		]
		if self.n_dofs == 3:
			self.field_value_names.append('displ_z')

		self.constitutive_model = constitutive_model
		self.formulation = formulation
		self.var_name_to_method = {
			'displacement': {
				'method': nodal_pp(self.field_values),
				'names' : self.field_value_names
			},
			'element_cauchy_stress': {
				'method': element_pp(self.element_quantity_from_func(
					self.constitutive_model.cauchy_stress, with_props=True
				)),
				'names' : full_tensor_names('cauchy_stress') 
			},
			'element_displacement_gradient': {
				'method': element_pp(self.element_field_gradient),
				'names' : full_tensor_names('displ_grad')
			},
			'element_deformation_gradient': {
				'method': element_pp(self.element_quantity_from_func(
					lambda x: x
				)),
				'names' : full_tensor_names('deformation_gradient')
			},
			'element_invariants': {
				'method': element_pp(self.element_quantity_from_func(
					self.constitutive_model.invariants
				)),
				'names' : ['I1', 'I2', 'I3']
			},
			'element_pk1_stress': {
				'method': element_pp(self.element_quantity_from_func(
					self.constitutive_model.pk1_stress, with_props=True
				)),
				'names' : full_tensor_names('pk1_stress')
			}
		}

	# TODO move these to base weakform class if possible?
	def element_field_gradient(self, params, domain, t):
		us = vmap(self.field_values, in_axes=(None, 0, None))(
			params.fields, domain.coords, t
		)
		# grads = compute_field_gradient(domain.fspace, us, domain.coords)
		# grads = domain.fspace.compute_field_gradient(domain.coords, us, domain.conns)
		grads = jax.vmap(domain.fspace.compute_field_gradient, in_axes=(0, 0))(
			us[domain.conns, :], domain.coords[domain.conns, :]
		)
		modify_field_grad = self.formulation.modify_field_gradient
		func = lambda grad: modify_field_grad(grad).flatten()
		return vmap(vmap(func))(grads)

	def element_quantity_from_func(self, func: Callable, with_props: Optional[bool] = False):
		if with_props:
			def new_func(params, domain, t):
				us = vmap(self.field_values, in_axes=(None, 0, None))(
					params.fields, domain.coords, t
				)
				props = params.properties()
				# grads = compute_field_gradient(domain.fspace, us, domain.coords)
				# grads = domain.fspace.compute_field_gradient(domain.coords, us, domain.conns)
				grads = jax.vmap(domain.fspace.compute_field_gradient, in_axes=(0, 0))(
					us[domain.conns, :], domain.coords[domain.conns, :]
				)
				modify_field_grad = self.formulation.modify_field_gradient
				grads = vmap(vmap(modify_field_grad))(grads)
				Fs = vmap(vmap(lambda grad: grad + jnp.eye(3)))(grads)
				temp_func = lambda F: func(F, props).flatten()
				return vmap(vmap(temp_func))(Fs)
		else:
			def new_func(params, domain, t):
				us = vmap(self.field_values, in_axes=(None, 0, None))(
					params.fields, domain.coords, t
				)
				# grads = compute_field_gradient(domain.fspace, us, domain.coords)
				# grads = domain.fspace.compute_field_gradient(domain.coords, us, domain.conns)
				grads = jax.vmap(domain.fspace.compute_field_gradient, in_axes=(0, 0))(
					us[domain.conns, :], domain.coords[domain.conns, :]
				)
				modify_field_grad = self.formulation.modify_field_gradient
				grads = vmap(vmap(modify_field_grad))(grads)
				Fs = vmap(vmap(lambda grad: grad + jnp.eye(3)))(grads)
				temp_func = lambda grad: func(grad).flatten()
				return vmap(vmap(temp_func))(Fs)
		return new_func

	def energy(self, x_el, u_el, N, grad_N, JxW, props):
		
		# caclulate quadrature level fields
		grad_u_q = self.formulation.modify_field_gradient((grad_N.T @ u_el).T)
		F_q = grad_u_q + jnp.eye(3)
		W = self.constitutive_model.energy(F_q, props)
		return JxW * W

	# trying out enforcing detF = 1
	def nodal_incompressibility_constraint(self, field_network, x, t):
		grad_u = self.field_gradients(field_network, x, t)
		grad_u = self.formulation.modify_field_gradient(grad_u)
		F = grad_u + jnp.eye(3)
		return (jnp.linalg.det(F) - 1.)**2

	def quadrature_incompressibility_constraint(self, x_el, u_el, N, grad_N, JxW, props):
		grad_u_q = self.formulation.modify_field_gradient((grad_N.T @ u_el).T)
		F_q = grad_u_q + jnp.eye(3)
		J_q = self.constitutive_model.jacobian(F_q)
		return 0.5 * JxW * (J_q - 1.)**2

	# TODO good code below just commenting out for now.
	def pk1_stress_weak_form(self, x_el, u_el, N, grad_N, JxW, props):
		grad_u_q = self.formulation.modify_field_gradient((grad_N.T @ u_el).T)
		F_q = grad_u_q + jnp.eye(3)
		P = jax.grad(self.constitutive_model.energy)(F_q, props)
		return P

	def strong_form_residual(self, params, x, t):
		div_stress = jax.jacfwd(self.pk1_stress, argnums=1)(params, x, t)
		div_stress = jnp.trace(div_stress, axis1=1, axis2=2)
		return div_stress

	def strong_form_neumann_bc(self, params, x, t, n):
		P = self.pk1_stress(params, x, t)
		# P = self.formulation.extract_stress(P)
		# n = jnp.hstack((n, jnp.zeros(1, dtype=jnp.float64)))
		traction = P @ n
		return traction

	def pk1_stress(self, params, x, t):
		field_network, props = params
		grad_u = self.field_gradients(field_network, x, t)
		# grad_u = self.formulation.modify_field_gradient(grad_u)
		# F = grad_u + jnp.eye(3)
		F = grad_u + jnp.eye(2)

		def const_model(F, props):
			props = props()
			K, G = props[0], props[1]
			# kinematics
			C = F.T @ F
			# J = self.jacobian(F)
			J = jnp.linalg.det(F)
			# I_1_bar = jnp.trace(jnp.power(J, -2. / 3.) * C)
			I_1_bar = jnp.trace(1. / jnp.square(jnp.sqrt(J)) * C)

			# constitutive
			W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
			W_dev = 0.5 * G * (I_1_bar - 2.)
			return W_vol + W_dev
		P = jax.grad(const_model, argnums=0)(F, props)
		# P = jax.grad(self.constitutive_model.energy)(F, props)
		return P
