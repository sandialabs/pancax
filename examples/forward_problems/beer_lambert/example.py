from jax.lax import stop_gradient
from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4.e')
logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 10.0, 1)
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
physics = BeerLambertLaw(jnp.array([0., 1.]), 1.e3)

##################
# bcs
##################
def bc_func(x, t, z):
  x, y = x[0], x[1]
  u = z

  I0 = 1.
  I0 = I0 * jnp.exp(-(x / 50e-6)**2)

  u = u.at[0].set((y + 0.0005) * u[0] + I0 * y)
  return u

# physics = physics.update_dirichlet_bc_func(bc_func)

ics = [
]
essential_bcs = [
  EssentialBC('nodeset_3', 0, lambda x, t: 10. * jnp.exp(-(x[0] / 100e-6)**2))
  # EssentialBC('nodeset_3', 0, lambda x, t: 1.)
]
natural_bcs = [
]

##################
# problem setup
##################
forward_problem = ForwardProblem(domain, physics, ics, essential_bcs, natural_bcs)

##################
# ML setup
##################
field = Field(forward_problem, key, ensure_positivity=True)
params = FieldPhysicsPair(field, forward_problem.physics)
loss_function = CombineLossFunctions(
  StrongFormResidualLoss(1.0),
  # ResidualMSELoss(1.),
  DirichletBCLoss(1.0e2)
)
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True, transition_steps=5000, decay_rate=0.95)
opt_st = opt.init(params)

for epoch in range(100000):
  params, opt_st, loss = opt.step(params, forward_problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    print(params.physics.x_mins)

##################
# post-processing
##################
pp.init(forward_problem, 'output.e',
  node_variables=['field_values']        
)
pp.write_outputs(params, forward_problem)
pp.close()
