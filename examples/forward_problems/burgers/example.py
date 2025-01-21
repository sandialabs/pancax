from pancax import *
# import pancax
# import pancax.domains_new
# import pancax.domains_new.variational_domain
# import pancax.physics_new
# import pancax.problems.forward_problem

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
times = jnp.linspace(0.0, 10.0, 21)
domain = CollocationDomain(mesh_file, times)
print(domain)
##################
# physics setup
##################
physics = BurgersEquation()

##################
# bcs
##################
def bc_func(x, t, z):
  x, y = x[0], x[1]
  # return (0.5 + x) * (0.5 - x) * z
  u = z
  u = u.at[0].set((0.5 + x) * (0.5 - x) * u[0])
  u = u.at[1].set((0.5 + x) * (0.5 - x) * u[1])

  # u = u.at[1].set()
  return u

def ic_func(x):
  x, y = x[0], x[1]
  return jnp.array([jnp.exp(-x**2 / 2.), 0.])

physics = physics.update_dirichlet_bc_func(bc_func)

ics = [
  ic_func
]
essential_bcs = [
  EssentialBC('nodeset_2', 0, lambda x, t: 0.0),
  EssentialBC('nodeset_4', 0, lambda x, t: 0.0),
  EssentialBC('nodeset_2', 1, lambda x, t: 0.0),
  EssentialBC('nodeset_4', 1, lambda x, t: 0.0),
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
n_dims = domain.coords.shape[1]
field = MLP(n_dims + 1, physics.n_dofs, 50, 5, jax.nn.tanh, key)
# props = FixedProperties([])
params = FieldPhysicsPair(field, forward_problem.physics)

print(forward_problem.physics.x_mins)
print(forward_problem.physics.x_maxs)

loss_function = EnergyLoss()
# loss_function_2 = StrongFormResidualLoss()
loss_function_2 = CombineLossFunctions(
  StrongFormResidualLoss(1.0),
  ICLossFunction(1.0),
  # DirichletBCLoss(1.0)
)
opt = Adam(loss_function_2, learning_rate=1e-3, has_aux=True)
opt_st = opt.init(params)
# # print(opt_st)

for epoch in range(25000):
  params, opt_st, loss = opt.step(params, forward_problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)

##################
# post-processing
##################
pp.init(forward_problem, 'output.e',
  node_variables=['field_values']        
)
pp.write_outputs(params, forward_problem)
pp.close()
