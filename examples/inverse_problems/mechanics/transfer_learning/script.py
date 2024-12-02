from pancax import *


##################
# for reproducibility
##################
key = random.key(100)

##################
# file management
##################
full_field_data_file = find_data_file('neohookean_50_bulk_modulus_100_full_field.csv')
global_data_file = find_data_file('neohookean_50_bulk_modulus_100_global_data.csv')
mesh_file = find_mesh_file('2holes.g')
logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 'z', 't'], ['u_x', 'u_y', 'u_z'])
# the 5 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 't', 'u_y', 'f_y',
  mesh_file, 5, 'y', # these inputs specify where to measure reactions
  n_time_steps=11, # the number of time steps for inverse problems is specified here
  plotting=True
)

##################
# domain setup
##################
times = jnp.linspace(0., 1., 11)
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
model = NeoHookean(
  bulk_modulus=100.0,
  shear_modulus=BoundedProperty(0., 5., key=key)
)
physics = SolidMechanics(model, ThreeDimensional())

##################
# ics/bcs setup
##################
ics = [
]
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=0.5, length=1.0, direction='y', n_dimensions=3
)
physics = physics.update_dirichlet_bc_func(essential_bc_func)
essential_bcs = [
  EssentialBC('nset_3', 0),
  EssentialBC('nset_3', 1),
  EssentialBC('nset_3', 2),
  EssentialBC('nset_5', 0),
  EssentialBC('nset_5', 1),
  EssentialBC('nset_5', 2) 
]
natural_bcs =[
]

##################
# problem setup
##################
problem = InverseProblem(domain, physics, ics, essential_bcs, natural_bcs, field_data, global_data)
print(problem)

##################
# ML setup
##################
field_network = MLP(3 + 1, 3, 100, 3, jax.nn.tanh, key)
params = FieldPropertyPair(field_network, problem.physics)

##################
# train network
##################
loss_function_1 = EnergyLoss()
loss_function_2 = CombineLossFunctions(
  EnergyResidualAndReactionLoss(energy_weight=1.0, residual_weight=0.001e9, reaction_weight=10.0e9),
  FullFieldDataLoss(weight=1.0e9),
  # with_props=True 
)

# filter_spec_1 = lambda p: p.field_network
# filter_spec_2 = lambda p: p.properties 
# filter_spec_1 = params.freeze_props_filter()
# filter_spec_2 = params.freeze_fields_filter()

opt_1 = Adam(loss_function_1, learning_rate=1.0e-3, has_aux=True)#, filter_spec=filter_spec_1)
opt_2 = Adam(loss_function_2, learning_rate=1.0e-3, has_aux=True)#, transition_steps=2500, filter_spec=filter_spec_2)

opt_1_st = opt_1.init(params)
opt_2_st = opt_2.init(params)

# pretrain
for epoch in range(2500):
  params_new, opt_1_st, loss = opt_1.step(params, problem, opt_1_st)

  # freeze props parameter here
  params = eqx.tree_at(lambda p: p.properties, params_new, params.properties)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    print(params.properties.constitutive_model.shear_modulus())

# now do real training
for n in range(1000):
  # iterate on field  parameters
  for epoch in range(250):
    params_new, opt_1_st, loss = opt_1.step(params, problem, opt_1_st)  

    # freeze props parameter here
    params = eqx.tree_at(lambda p: p.properties, params_new, params.properties)

  # iterate on props
  for epoch in range(50):
    params_new, opt_2_st, loss = opt_2.step(params, problem, opt_2_st)  

    # freeze field parameter here
    params = eqx.tree_at(lambda p: p.fields, params_new, params.fields)

  print(epoch)
  print(loss)
  print(params.properties.constitutive_model.shear_modulus())

##################
# post-processing
##################
pp.init(problem, 'output.e', 
  node_variables=[
    # 'displacement'
    'field_values'
  ], 
  element_variables=[]
)
pp.write_outputs(params, problem)
pp.close()
# trainer_1 = Trainer(
#   problem, opt_1, 
#   # output_node_variables=['displacement'], 
#   output_node_variables=['field_values'],
#   log_every=2500, 
#   output_every=100000
# )
# trainer_2 = Trainer(
#   problem, opt_2, 
#   # output_node_variables=['displacement'], 
#   output_node_variables=['field_values'],
#   log_file='props_log.log',
#   log_every=10,
#   output_every=1000000000
# )
# opt_st_2 = trainer_2.init(params)

# pre-train on initial prop guesses
# print('Pre-training')
# params = trainer_1.train(params, 25000)
# for n in range(1000):
#   print(f'Iteration = {n}')
#   params = trainer_1.train(params, 5000)
#   for m in range(50):
#     params, opt_st_2 = trainer_2.step(params, opt_st_2)
