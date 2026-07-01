from pancax import *


##################
# for reproducibility
##################
key = random.key(100)

##################
# file management
##################
full_field_data_file = './data/dic_dataset_0.csv'
global_data_file = './data/mts_dataset.csv'
mesh_file = './mesh/mesh.g'
logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(
  full_field_data_file, 
  ['x', 'y', 'z', 't'], 
  ['u_x', 'u_y', 'u_z']
)
shifted_inputs = field_data.inputs
shifted_inputs = shifted_inputs.at[:, 1].set(shifted_inputs[:, 1] + 31.115)
shifted_inputs = shifted_inputs.at[:, 2].set(1.905) # to make z coords correct
field_data = eqx.tree_at(lambda x: x.inputs, field_data, shifted_inputs)


# the 5 below is for the nodeset id
global_data = GlobalData(
  global_data_file,
  'times', 'displacement', 'force',
  mesh_file, 2, 'y', # these inputs specify where to measure reactions
  n_time_steps=11, # the number of time steps for inverse problems is specified here
  plotting=True # for sanity check
)
# need to negate data
# new_reactions = global_data.reactions
global_data = eqx.tree_at(lambda x: x.outputs, global_data, -global_data.outputs)

# ##################
# # domain setup
# ##################
times = jnp.linspace(0., 1., 11)
domain = VariationalDomain(mesh_file, times)

# sanity check plots
field_data.plot_registration(domain)

# ##################
# # physics setup
# ##################
model = NeoHookean(
  bulk_modulus=980.0,
  shear_modulus=BoundedProperty(0.01, 10., key=key)
)
physics = SolidMechanics(model, ThreeDimensional())

##################
# ics/bcs setup
##################
ics = [
]
# essential_bc_func = UniaxialTensionLinearRamp(
#   final_displacement=0.5, length=1.0, direction='y', n_dimensions=3
# )
# physics = physics.update_dirichlet_bc_func(essential_bc_func)
essential_bcs = [
  EssentialBC('nset_bottom', 0),
  EssentialBC('nset_bottom', 1),
  EssentialBC('nset_bottom', 2),
  EssentialBC('nset_top', 0),
  EssentialBC('nset_top', 1, lambda x, t: -jnp.max(global_data.displacements) * t),
  EssentialBC('nset_top', 2) 
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
loss_function_1 = CombineLossFunctions(
  EnergyLoss(),
  DirichletBCLoss(weight=1.e4)
)
loss_function_2 = CombineLossFunctions(
  # EnergyResidualAndReactionLoss(energy_weight=1.0, residual_weight=0.001e9, reaction_weight=10.0e9),
  EnergyResidualAndReactionLoss(energy_weight=1.0, residual_weight=1.e9, reaction_weight=1.e9),
  DirichletBCLoss(1.e9),
  FullFieldDataLoss(weight=1.e9),
  # with_props=True 
)

opt_1 = Adam(loss_function_1, learning_rate=1.0e-3, has_aux=True)#, filter_spec=filter_spec_1)
opt_2 = Adam(loss_function_2, learning_rate=1.0e-3, has_aux=True)#, transition_steps=2500, filter_spec=filter_spec_2)

opt_1_st = opt_1.init(params)
opt_2_st = opt_2.init(params)

# pretrain
for epoch in range(25000):
  params_new, opt_1_st, loss = opt_1.step(params, problem, opt_1_st)

  # freeze props parameter here
  params = eqx.tree_at(lambda p: p.properties, params_new, params.properties)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    print(params.properties.constitutive_model.shear_modulus())

params.serialise('pretrained', 0)
# params = eqx.tree_deserialise_leaves('pretrained.eqx_0000000.eqx', params)

pp.init(problem, 'output.e', 
  node_variables=[
    # 'displacement'
    'field_values'
  ], 
  element_variables=[]
)
pp.write_outputs(params, problem)
pp.close()

# now do real training
for n in range(10000):
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

  if n % 10 == 0:
    params.serialise(f'restart', n)

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
