from .base_kernel import StrongFormPhysicsKernel, WeakFormPhysicsKernel
from .base_kernel import standard_pp
from typing import Callable, Optional
import jax.numpy as jnp


class Poisson(StrongFormPhysicsKernel, WeakFormPhysicsKernel):
	n_dofs = 1
	field_value_names = ['u']
	use_delta_pinn: bool

	def __init__(
		self, 
		mesh_file, 
		bc_func: Callable, 
		f: Callable,
		use_delta_pinn: Optional[bool] = False
	):
		super().__init__(mesh_file, bc_func, use_delta_pinn)
		self.f = f
		self.var_name_to_method = standard_pp(self)

	def energy(self, x_el, u_el, N, grad_N, JxW, props):
		# caclulate quadrature level fields
		x_q = jnp.dot(N, x_el)
		u_q = jnp.dot(N, u_el)
		grad_u_q = (grad_N.T @ u_el).T

		# kernel specific stuff here
		f_q = self.f(x_q)
		pi_q = 0.5 * jnp.dot(grad_u_q, grad_u_q.T) - f_q * u_q
		return JxW * pi_q

	def strong_form_residual(self, params, x, t):
		field_network, props = params
		delta_u = self.field_laplacians(field_network, x, t)[0]
		f = self.f(x)
		return -delta_u - f

	def strong_form_neumann_bc(self, params, x, t, n):
		field_network, props = params
		grad_u = self.field_gradients(field_network, x, t)
		return -jnp.dot(grad_u, n)
