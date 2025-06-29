from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
full_field_data_file = find_data_file('data_full_field.csv')
global_data_file = find_data_file('data_global_data.csv')
mesh_file = find_mesh_file('mesh.g')
logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 't'], ['u_x', 'u_y'])
# the 4 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 't', 'u_x', 'f_x',
  mesh_file, 4, 'x', # these inputs specify where to measure reactions
  n_time_steps=11, # the number of time steps for inverse problems is specified here
  plotting=True
)

##################
# domain setup
##################
times = jnp.linspace(0.0, 1.0, len(global_data.outputs))
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
model = NeoHookean(
  bulk_modulus=0.833,
  # bulk_modulus=BoundedProperty(0.01, 5., key=key),
  shear_modulus=BoundedProperty(0.01, 5., key=key)
)
# model = Gent(
#   bulk_modulus=0.833,
#   shear_modulus=BoundedProperty(0.01, 5., key=key),
#   Jm_parameter=BoundedProperty(1.5, 10., key=key)
# )
physics = SolidMechanics(model, PlaneStrain())

##################
# ics/bcs
##################
ics = [
]
dirichlet_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 0]), 
  length=1.0, direction='x', n_dimensions=2
)
physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)
dirichlet_bcs = [
  DirichletBC('nodeset_2', 0), # left edge fixed in x
  DirichletBC('nodeset_2', 1), # left edge fixed in y
  DirichletBC('nodeset_4', 0), # right edge prescribed in x
  DirichletBC('nodeset_4', 1)  # right edge fixed in y
]
neumann_bcs = [
]

##################
# problem setup
##################
# problem = InverseProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs, field_data, global_data)
problem = InverseProblem(domain, physics, field_data, global_data, ics, dirichlet_bcs, neumann_bcs)

# print(problem)

##################
# ML setup
##################
# n_dims = domain.coords.shape[1]
# field = MLP(n_dims + 1, physics.n_dofs, 50, 5, jax.nn.tanh, key)
# params = FieldPropertyPair(field, problem.physics)
params = Parameters(problem, key, seperate_networks=True, network_type=ResNet)

# print(params)
# loss_function = EnergyResidualAndReactionLoss()
loss_function = CombineLossFunctions(
  EnergyResidualAndReactionLoss(residual_weight=250.0e9, reaction_weight=250.0e9),
  FullFieldDataLoss(weight=10.0e9),
  # with_props=True
)
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=5000)

##################
# Training
##################
opt_st = opt.init(params)

for epoch in range(100000):
  params, opt_st, loss = opt.step(params, problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    print(params.physics.constitutive_model.bulk_modulus)
    print(params.physics.constitutive_model.shear_modulus())
    # print(params.physics.constitutive_model.Jm_parameter())
