from pancax import *

##################
# for reproducibility
##################
# key = random.key(100)
key = random.PRNGKey(100)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4.g')
# logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 1.0, 2)
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
physics = Poisson(lambda x: 2 * jnp.pi**2 * jnp.sin(2. * jnp.pi * x[0]) * jnp.sin(2. * jnp.pi * x[1]))

##################
# bcs
##################
def bc_func(x, t, z):
  x, y = x[0], x[1]
  return x * (1. - x) * y * (1. - y) * z

# physics = physics.update_dirichlet_bc_func(bc_func)

ics = [
]
dirichlet_bcs = [
  DirichletBC('nset_1', 0),
  DirichletBC('nset_2', 0),
  DirichletBC('nset_3', 0),
  DirichletBC('nset_4', 0),
]
neumann_bcs = [
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs)

##################
# ML setup
##################
loss_function = EnergyLoss()
params = Parameters(problem, key, dirichlet_bc_func=bc_func)

# pre-train with Adam
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt, opt_st = opt.init(params)
for epoch in range(2500):
  params, opt_st, loss = opt.step(params, opt_st, problem)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)



# # switch to LBFGS
# params, static = eqx.partition(params, eqx.is_inexact_array)

# def loss_func(params):
#   params = eqx.combine(params, static)
#   loss, aux = loss_function(params, problem)
#   return loss

# opt = optax.lbfgs(memory_size=1)
# opt_st = opt.init(params)
# value_and_grad = jax.jit(optax.value_and_grad_from_state(loss_func))
# for _ in range(200):
#   value, grad = value_and_grad(params, state=opt_st)
#   updates, opt_st = opt.update(
#      grad, opt_st, params, value=value, grad=grad, value_fn=loss_func
#   )
#   params = optax.apply_updates(params, updates)
#   print('Objective function: {:.2E}'.format(value))

##################
# post-processing
##################
# params = eqx.combine(params, static)
pp.init(params, problem, 'output.e',
  node_variables=['field_values']        
)
pp.write_outputs(params, problem)
pp.close()
