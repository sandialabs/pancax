from pancax import *

##################
# for reproducibility
##################
key = random.PRNGKey(10)
# key = random.split(key, 8)

##################
# file management
##################
full_field_data_file = find_data_file('full_field_data.csv')
global_data_file = find_data_file('global_data.csv')
# mesh_file = find_mesh_file('mesh.g')
# mesh_file = 'data/2holes.g'
mesh_file = os.path.join(Path(__file__).parent, "data", "2holes.g")

history = HistoryWriter('history.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 'z', 't'], ['displ_x', 'displ_y', 'displ_z'])
# the 4 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 'times', 'disps', 'forces',
  mesh_file, 5, 'y', # these inputs specify where to measure reactions
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
  bulk_modulus=10.,
  shear_modulus=BoundedProperty(0.01, 5., key=key)
  # shear_modulus=0.855
)
# model = InputPolyConvexPotential(
#   bulk_modulus=10., key=key
# )
physics = SolidMechanics(model, ThreeDimensional())

##################
# ics/bcs
##################
ics = [
]
dirichlet_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 1]), 
  length=1.0, direction='y', n_dimensions=3
)
dirichlet_bcs = [
  DirichletBC('nodeset_3', 0), # left edge fixed in x
  DirichletBC('nodeset_3', 1), # left edge fixed in y
  DirichletBC('nodeset_3', 2),
  DirichletBC('nodeset_5', 0), # right edge prescribed in x
  DirichletBC('nodeset_5', 1), # right edge fixed in y
  DirichletBC('nodeset_5', 2)
]
neumann_bcs = [
]

##################
# problem setup
##################
problem = InverseProblem(domain, physics, field_data, global_data, ics, dirichlet_bcs, neumann_bcs)

# print(problem)

##################
# ML setup
##################
params = Parameters(
  problem, key,
  dirichlet_bc_func=dirichlet_bc_func
  # seperate_networks=True, 
  # network_type=ResNet
)
print(params)
physics_and_global_loss = EnergyResidualAndReactionLoss(
  residual_weight=50.e9, reaction_weight=250.e9
)
full_field_data_loss = FullFieldDataLoss(weight=500.e9)

def loss_function(params, problem, inputs, outputs):
  loss_1, aux_1 = physics_and_global_loss(params, problem)
  loss_2, aux_2 = full_field_data_loss(params, problem, inputs, outputs)
  aux_1.update(aux_2)
  return loss_1 + loss_2, aux_1

loss_function = UserDefinedLossFunction(loss_function)
# loss_function = EnergyLoss()

opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=10000)

##################
# Training
##################
opt, opt_st = opt.init(params)

dataloader = FullFieldDataLoader(problem.field_data)
for epoch in range(100000):
  for inputs, outputs in dataloader.dataloader(512):
    params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs)
    
    # # params.physics.network
    # new_model = params.physics.constitutive_model.parameter_enforcement()
    # # physics = SolidMechanics(model, PlaneStrain())
    # new_physics = eqx.tree_at(lambda x: x.constitutive_model, params.physics, new_model)
    # params = eqx.tree_at(lambda x: x.physics, params, new_physics)

  if epoch % 10 == 0:
    print(epoch)
    print(loss)
    # print(params.physics.constitutive_model)
    # print(params.physics.constitutive_model.shear_modulus.prop_min)
    # print(params.physics.constitutive_model.shear_modulus.prop_max)

    print(params.physics.constitutive_model.shear_modulus())
