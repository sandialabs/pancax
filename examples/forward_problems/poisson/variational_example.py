from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4.g')
logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 0.0, 1)
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

physics = physics.update_dirichlet_bc_func(bc_func)

ics = [
]
essential_bcs = [
  EssentialBC('nset_1', 0),
  EssentialBC('nset_2', 0),
  EssentialBC('nset_3', 0),
  EssentialBC('nset_4', 0),
]
natural_bcs = [
]

##################
# problem setup
##################
problem = Problem(domain, physics, ics, essential_bcs, natural_bcs)

##################
# ML setup
##################
n_dims = domain.coords.shape[1]
field = MLP(n_dims + 1, physics.n_dofs, 50, 3, jax.nn.tanh, key)
params = FieldPropertyPair(field, problem.physics)

loss_function = EnergyLoss()
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt_st = opt.init(params)

for epoch in range(5000):
  params, opt_st, loss = opt.step(params, problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)

##################
# post-processing
##################
pp.init(problem, 'output.e',
  node_variables=['field_values']        
)
pp.write_outputs(params, problem)
pp.close()
