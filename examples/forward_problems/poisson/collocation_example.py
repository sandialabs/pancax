from pancax import *

##################
# for reproducibility
##################
key = random.PRNGKey(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4.g')
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 0.0, 1)
domain = CollocationDomain(mesh_file, times)

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

ics = [
]
essential_bcs = [
  DirichletBC('nset_1', 0),
  DirichletBC('nset_2', 0),
  DirichletBC('nset_3', 0),
  DirichletBC('nset_4', 0),
]
natural_bcs = [
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, essential_bcs, natural_bcs)

##################
# ML setup
##################
params = Parameters(problem, key, dirichlet_bc_func=bc_func, network_type=ResNet)

def loss_function(params, problem, inputs, outputs):
  field, physics, state = params
  residuals = jax.vmap(physics.strong_form_residual, in_axes=(None, 0, 0))(
    field, inputs[:, 0:2], inputs[:, 2:3]
  )
  return jnp.square(residuals - outputs).mean(), dict(nothing=0.0)

loss_function = UserDefinedLossFunction(loss_function)

opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt, opt_st = opt.init(params)

dataloader = CollocationDataLoader(problem.domain, num_fields=1)
for epoch in range(50000):
  for inputs, outputs in dataloader.dataloader(512):
    params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)

# ##################
# # post-processing
# ##################
# pp.init(params, problem, 'output.e',
#   node_variables=['field_values']        
# )
# pp.write_outputs(params, problem)
# pp.close()

# # import pyvista as pv
# # exo = pv.read('output.e')[0][0]
# # exo.set_active_scalars('u')
# # exo.plot(show_axes=False, cpos='xy', show_edges=True)