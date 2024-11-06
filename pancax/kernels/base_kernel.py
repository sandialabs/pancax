from abc import ABC
from abc import abstractmethod
from jax import hessian, jacfwd, vmap
from jax import numpy as jnp
from pancax.fem import read_exodus_mesh
from pancax.fem.function_space import compute_field_gradient
from typing import Callable, Dict, List, Union
import equinox as eqx


def vector_names(base_name: str, dim: int):
	pass


def full_tensor_names(base_name: str):
	"""
	Provides a full list of tensorial variable component names
	:param base_name: base name for a tensor variable e.g. base_name_xx
	"""
	return [
		f'{base_name}_xx', f'{base_name}_xy', f'{base_name}_xz', 
		f'{base_name}_yx', f'{base_name}_yy', f'{base_name}_yz', 
		f'{base_name}_zx', f'{base_name}_zy', f'{base_name}_zz'
	]


def full_tensor_names_2D(base_name: str):
	"""
	Provides a full list of tensorial variable component names
	:param base_name: base name for a tensor variable e.g. base_name_xx
	"""
	return [
		f'{base_name}_xx', f'{base_name}_xy', 
		f'{base_name}_yx', f'{base_name}_yy'
	]

# TODO improve this
def element_pp(func, has_props=False, jit=True):
	"""
	:param func: Function to use for an element property output variable
	:param has_props: Whether or not this function need properties
	:param jit: Whether or not to jit this function
	"""
	if jit:
		return eqx.filter_jit(func)
	else:
		return func


def nodal_pp(func, has_props=False, jit=True):
	"""
	:param func: Function to use for a nodal property output variable
	:param has_props: Whether or not this function need properties
	:param jit: Whether or not to jit this function
	"""
	if has_props:
		# new_func = lambda p, d, t: vmap(
		# 	func, in_axes=(None, 0, None, None)
		# )(p.fields, d.coords, t, p.properties)
		new_func = lambda p, d, t: vmap(
			func, in_axes=(None, 0, None)
		)(p, d.coords, t)
	else:
		new_func = lambda p, d, t: vmap(
			func, in_axes=(None, 0, None)
		)(p.fields, d.coords, t)
	
	if jit:
		return eqx.filter_jit(new_func)
	else:
		return new_func


# make a standard pp method that just has nodal fields, element grads, etc.
def standard_pp(physics):
	d = {
		'field_values': {
			'method': nodal_pp(physics.field_values),
			'names' : physics.field_value_names
		}
	}
	return d


class PhysicsKernel(ABC):
	n_dofs: int
	field_value_names: List[str]
	bc_func: Callable # further type this guy
	var_name_to_method: Dict[str, Dict[str, Union[Callable, List[str]]]] = {}
	use_delta_pinn: bool

	def __init__(
		self, 
		mesh_file, 
		bc_func, 
		use_delta_pinn
	) -> None:
		self.bc_func = bc_func
		self.use_delta_pinn = use_delta_pinn

		# currently a dumb setup
		mesh = read_exodus_mesh(mesh_file)

		self.x_mins = jnp.min(mesh.coords, axis=0)
		self.x_maxs = jnp.max(mesh.coords, axis=0)

		print('Bounding box for mesh:')
		print(f'  x_mins = {self.x_mins}')
		print(f'  x_maxs = {self.x_maxs}')

		if self.use_delta_pinn:
			def field_basis(self, basis_network, x, t, v):
				inputs = jnp.hstack((v, t))
				z = basis_network(inputs)
				return z

			def field_values(field_network, x, t, v):
				# TODO assume here for now that modes are normalized
				# v_temp = (v - jnp.min(v, axis=0)) / (jnp.max(v, axis=0) - jnp.min(v, axis=0))
				# inputs = jnp.hstack((v_temp, t))
				inputs = jnp.hstack((v, t))
				z = field_network(inputs)
				u = self.bc_func(x, t, z)
				return u
		else:
			def field_basis(self, basis_network, x, t):
				x = (x - self.x_mins) / (self.x_maxs - self.x_mins)
				inputs = jnp.hstack((x, t))
				z = basis_network(inputs)
				return z

			def field_values(field_network, x, t):
				x_temp = (x - self.x_mins) / (self.x_maxs - self.x_mins)
				inputs = jnp.hstack((x_temp, t))
				z = field_network(inputs)
				u = self.bc_func(x, t, z)
				return u
	
		self.field_basis = field_basis
		self.field_values = field_values


class StrongFormPhysicsKernel(PhysicsKernel):
	n_dofs: int
	field_value_names: List[str]
	bc_func: Callable # further type this guy
	use_delta_pinn: bool
	
	def __init__(self, mesh_file, bc_func, use_delta_pinn) -> None:
		# if use_delta_pinn:
		# 	raise ValueError('DeltaPINNs are currently not supported with collocation PINNs.')
		super().__init__(mesh_file, bc_func, use_delta_pinn)

	def field_gradients(self, field_network, x, t):
		return jacfwd(self.field_values, argnums=1)(field_network, x, t)

	def field_hessians(self, field_network, x, t):
		return hessian(self.field_values, argnums=1)(field_network, x, t)

	def field_laplacians(self, field_network, x, t):
		return jnp.trace(self.field_hessians(field_network, x, t), axis1=1, axis2=2)

	def field_time_derivatives(self, field_network, x, t):
		return jacfwd(self.field_values, argnums=2)(field_network, x, t)

	def field_second_time_derivatives(self, field_network, x, t):
		return jacfwd(self.field_time_derivatives, argnums=2)(field_network, x, t)

	@abstractmethod
	def strong_form_neumann_bc(self, params, x, t, n):
		pass

	@abstractmethod
	def strong_form_residual(self, params, x, t):
		pass


class WeakFormPhysicsKernel(PhysicsKernel):
	n_dofs: int
	field_value_names: List[str]
	bc_func: Callable # further type this guy
	use_delta_pinn: bool
	
	def __init__(self, mesh_file, bc_func, use_delta_pinn) -> None:
		super().__init__(mesh_file, bc_func, use_delta_pinn)

	def element_field_gradient(self, params, domain, t):
		us = self.field_values(params, domain.coords, t)
		grads = compute_field_gradient(domain.fspace, us, domain.coords)
		return grads

	# newer methods
	def element_quantity(self, f, fspace, x_el, u_el, props):
		Ns = fspace.shape_function_values(x_el)
		grad_Ns = fspace.shape_function_gradients(x_el)
		JxWs = fspace.JxWs(x_el)
		return jnp.sum(vmap(f, in_axes=(None, None, 0, 0, 0, None))(x_el, u_el, Ns, grad_Ns, JxWs, props))

	def element_quantity_grad(self, f, fspace, x_el, u_el, props):
		return jacfwd(self.element_quantity, argnums=3)(f, fspace, x_el, u_el, props)

	def element_quantity_hessian(self, f, fspace, x_el, u_el, props):
		return hessian(self.element_quantity, argnums=3)(f, fspace, x_el, u_el, props)

	@abstractmethod
	def energy(self, x_el, u_el, N, grad_N, JxW, props):
		pass
