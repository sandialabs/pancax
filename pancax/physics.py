from jax import grad
from jax import hessian
from jax import jacfwd
from jax import value_and_grad
from jax import vmap
from pancax.fem import assemble_sparse_stiffness_matrix
from pancax.timer import Timer
import jax.numpy as jnp
import numpy as onp


# TODO remove above and start using this method below
def element_quantity(f, x, u, conn, N, grad_N, JxW, props):
	x_el = x[conn, :]
	u_el = u[conn, :]
	return vmap(f, in_axes=(None, None, 0, 0, 0, None))(x_el, u_el, N, grad_N, JxW, props)
	

def element_quantity_new(f, fspace, x_el, u_el, props):
	Ns = fspace.shape_function_values(x_el)
	grad_Ns = fspace.shape_function_gradients(x_el)
	JxWs = fspace.JxWs(x_el)
	return jnp.sum(vmap(f, in_axes=(None, None, 0, 0, 0, None))(x_el, u_el, Ns, grad_Ns, JxWs, props))


element_quantity_grad = jacfwd(element_quantity_new, argnums=3)
element_quantity_hessian = hessian(element_quantity_new, argnums=3)


def potential_energy(domain, us, props):

	# unpack some stuff
	us = us[domain.conns, :]
	coords = domain.coords[domain.conns, :]
	func = domain.physics.energy

	pis = vmap(element_quantity_new, in_axes=(None, None, 0, 0, None))(
		func, domain.fspace, coords, us, props
	)
	return jnp.sum(pis)


def residual_mse(domain, us, props):
	# unpack some stuff
	us = us[domain.conns, :]
	coords = domain.coords[domain.conns, :]
	kernel = domain.physics

	func = element_quantity_grad
	fs = vmap(func, in_axes=(None, None, 0, 0, None))(
		kernel.energy, domain.fspace, coords, us, props
	)
	def vmap_func(f, unknowns):
		return jnp.square(jnp.where(unknowns, f, 0)).mean()

	return vmap(vmap_func, in_axes=(0, 0))(fs, domain.dof_manager.isUnknown[domain.conns, :]).mean()


def mass_matrix(domain, us, props):
	with Timer('mass matrix assembly'):
		us = us[domain.conns, :]
		coords = domain.coords[domain.conns, :]
		kernel = domain.physics

		func = element_quantity_hessian
		fs = vmap(func, in_axes=(None, None, 0, 0, None))(
			kernel.kinetic_energy, domain.fspace, coords, us, props
		)
		return assemble_sparse_stiffness_matrix(
			onp.asarray(fs), onp.asarray(domain.conns), domain.dof_manager
		)


def stiffness_matrix(domain, us, props):
	with Timer('stiffness matrix assembly'):
		us = us[domain.conns, :]
		coords = domain.coords[domain.conns, :]
		kernel = domain.physics

		func = element_quantity_hessian
		fs = vmap(func, in_axes=(None, None, 0, 0, None))(
			kernel.energy, domain.fspace, coords, us, props
		)
		return assemble_sparse_stiffness_matrix(
			onp.asarray(fs), onp.asarray(domain.conns), domain.dof_manager
		)


# TODO
def traction_energy():
  pass


# internal force methods
internal_force = grad(potential_energy, argnums=1)

# residual methods
residual = lambda x, y, z: jnp.linalg.norm(internal_force(x, y, z).flatten()[x.dof_manager.unknownIndices])

# combined methods
potential_energy_and_internal_force = value_and_grad(potential_energy, argnums=1)


def potential_energy_and_residual(domain, us, props):
	pi, f = potential_energy_and_internal_force(domain, us, props)
	return pi, jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])


def potential_energy_residual_and_reaction_force(domain, us, props):
	pi, f = potential_energy_and_internal_force(domain, us, props)
	R = jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])
	reaction = jnp.sum(f[domain.global_data.reaction_nodes, domain.global_data.reaction_dof])
	return pi, R, reaction


def strong_form_residual(params, domain, t):
	residuals = vmap(domain.physics.strong_form, in_axes=(None, 0, None))(params, domain.coords, t)
	residual = jnp.square(residuals.flatten()[domain.dof_manager.unknownIndices]).mean()
	return residual


# test
def nodal_incompressibility_constraint(params, domain, t):
	vals = vmap(domain.physics.nodal_incompressibility_constraint, in_axes=(None, 0, None))(
		params.fields, domain.coords, t
	)
	return jnp.sum(vals)
	# return domain.physics.nodal_incompressibility_constraint(params, domain, t)


def quadrature_incompressibility_constraint(domain, us, props):
	# unpack some stuff
	us = us[domain.conns, :]
	coords = domain.coords[domain.conns, :]
	func = domain.physics.quadrature_incompressibility_constraint

	# vmap over elements
	vals = vmap(element_quantity_new, in_axes=(None, None, 0, 0, None))(
		func, domain.fspace_centroid, coords, us, props
	)
	# return jnp.mean(vals)
	return jnp.sum(vals)

# quadrature_incompressibility_constraint_energy_and_residual = grad(
# 	quadrature_incompressibility_constraint, argnums=1
# )

# def quadrature_incompressibility_constraint_energy_and_residual(domain, us, props):
# 	loss, f = quadrature_incompressibility_constraint_energy_and_residual(domain, us, props)
# 	return loss, jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])


def incompressible_energy(domain, us, props):
	K = 100.0
	pi = potential_energy(domain, us, props)
	constraint = quadrature_incompressibility_constraint(domain, us, props)
	return pi + K * constraint

incompressible_internal_force = grad(incompressible_energy, argnums=1)
incompressible_energy_and_internal_force = value_and_grad(incompressible_energy, argnums=1)

def incompressible_energy_and_residual(domain, us, props):
	pi, f = incompressible_energy_and_internal_force(domain, us, props)
	return pi, jnp.linalg.norm(f.flatten()[domain.dof_manager.unknownIndices])
