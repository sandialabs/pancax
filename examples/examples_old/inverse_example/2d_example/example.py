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
# physics setup
##################
times = jnp.linspace(0.0, 1.0, len(global_data.outputs))
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 0]), 
  length=1.0, direction='x', n_dimensions=2
)
model = NeoHookeanFixedBulkModulus(bulk_modulus=0.833)
# model = NeoHookean()
formulation = PlaneStrain()
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, model, formulation)
essential_bcs = [
  EssentialBC('nodeset_2', 0), # left edge fixed in x
  EssentialBC('nodeset_2', 1), # left edge fixed in y
  EssentialBC('nodeset_4', 0), # right edge prescribed in x
  EssentialBC('nodeset_4', 1)  # right edge fixed in y
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
  EnergyResidualAndReactionLoss(alpha=250.0e9, beta=250.0e9),
  FullFieldDataLoss(essential_bc_func, weight=10.0e9),
  with_props=True
)
props = Properties(
  prop_mins=[0.1],
  prop_maxs=[10.0],
  key=key
)
network = MLP(3, 2, 50, 5, jax.nn.tanh, key)
params = FieldPropertyPair(network, props)

##################
# Training
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.init(params)
for epoch in range(250000):
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
  if epoch % 10000 == 0:
    pp.init(domain, 'output.e', 
      node_variables=[
        'displacement',
        'internal_force'
      ], 
      element_variables=[
        'element_cauchy_stress',
        'element_displacement_gradient',
        'element_deformation_gradient',
        'element_invariants',
        'element_pk1_stress'
      ]
    )
    pp.write_outputs(params, domain)
    pp.close()
