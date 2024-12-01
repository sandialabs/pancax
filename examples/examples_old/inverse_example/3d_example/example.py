from pancax import *
# parameters used to generate data with Sierra SM
# bulk modulus = 10.0
# A1 = 0.93074321
# P1 = -0.07673672
# B1 = 0.0
# Q1 = 0.5
# C1 = 0.10448312
# R1 = 1.71691036

##################
# for reproducibility
##################
# jax.config.update("jax_debug_nans", True) # will slow things down a lot
key = random.key(10)

##################
# file management
##################
# full_field_data_file = find_data_file('data_full_field.csv')
# global_data_file = find_data_file('global_data_with_origin.csv')
full_field_data_file = find_data_file('tension_template_300percent_full_field.csv')
global_data_file = find_data_file('tension_template_300percent_global_data.csv')
mesh_file = find_mesh_file('mesh_hex8_coarse.g')
logger = Logger('pinn.log', log_every=1000)
history = HistoryWriter('history.csv', log_every=1000, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 'z', 't'], ['u_x', 'u_y', 'u_z'])
# the 6 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 't', 'u_y', 'f_y',
  mesh_file, 5, 'y', # these inputs specify where to measure reactions
  n_time_steps=11, # the number of time steps for inverse problems is specified here
  plotting=True
)

##################
# physics setup
##################
times = global_data.times
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 1]), 
  length=1.0, direction='y', n_dimensions=3
)
model = Swanson4FixedBulkModulus(bulk_modulus=10.0, cutoff_strain=0.01)
formulation = ThreeDimensional()
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, model, formulation)
essential_bcs = [
  EssentialBC('nset_3', 0),
  EssentialBC('nset_3', 1),
  EssentialBC('nset_3', 2),
  EssentialBC('nset_5', 0),
  EssentialBC('nset_5', 1),
  EssentialBC('nset_5', 2)
]
natural_bcs = [
]
domain = InverseDomain(
  physics_kernel, essential_bcs, natural_bcs, mesh_file, times,
  field_data, global_data
)

##################
# ML setup
##################
loss_function = CombineLossFunctions(
  EnergyResidualAndReactionLoss(residual_weight=250.0e9, reaction_weight=250.0e9),
  FullFieldDataLoss(weight=100.0e9),
  with_props=True
)
props = Properties(
  prop_mins=[0.45485, -0.31492, 0.000002, 0.00172], 
  prop_maxs=[2.98489, -0.04736, 0.450380, 1.86427],
  key=key
)
network = MLP(4, 3, 50, 5, jax.nn.tanh, key)
params = FieldPropertyPair(network, props)

##################
# Training
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=2500)
opt_st = opt.init(params)
for epoch in range(1000000):
  params, opt_st, loss = opt.step(params, domain, opt_st)

  ##################
  # logging and history tracking
  ##################
  logger.log_loss(loss, epoch)
  history.write_history(loss, epoch)

  ##################
  # save restart file
  ##################
  if epoch % 10000 == 0:
    params.serialise('checkpoint', epoch)

  ##################
  # post-processing
  ##################
  if epoch % 25000 == 0:
    pp.init(domain, 'output.e', 
      node_variables=[
        'displacement',
        'internal_force'
      ], 
      element_variables=[
        # 'element_cauchy_stress',
        # 'element_displacement_gradient',
        # 'element_deformation_gradient',
        # 'element_invariants',
        # 'element_pk1_stress'
      ]
    )
    pp.write_outputs(params, domain)
    pp.close()
