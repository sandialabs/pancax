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
ics = [
]
dirichlet_bcs = [
  DirichletBC('nodeset_3', 0, lambda x, t: 10. * jnp.exp(-(x[0] / 100e-6)**2))
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
field = MLDirichletField(problem, key, ensure_positivity=True)
params = FieldPhysicsPair(field, problem.physics)
loss_function = ResidualMSELoss(1.)
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True, transition_steps=5000, decay_rate=0.95)
opt_st = opt.init(params)

for epoch in range(25000):
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
