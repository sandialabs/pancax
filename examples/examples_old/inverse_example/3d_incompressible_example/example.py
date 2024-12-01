from pancax import *

##################
# for reproducibility
##################
# jax.config.update("jax_debug_nans", True) # will slow things down a lot
key = random.key(10)

##################
# file management
##################
full_field_data_file = find_data_file('tension_template_full_field.csv')
global_data_file = find_data_file('tension_template_global_data.csv')
mesh_file = find_mesh_file('mesh_hex8_coarse.g')
logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history.csv', log_every=250, write_every=250)
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
# model = Swanson4FixedBulkModulus(bulk_modulus=10.0, cutoff_strain=0.01)
# model = NeoHookeanFixedBulkModulus(bulk_modulus=1500.0)
model = NeoHookeanFixedBulkModulus(bulk_modulus=100.0)
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
  EnergyResidualAndReactionLoss(alpha=250.0e9, beta=250.0e9),
  FullFieldDataLoss(essential_bc_func, weight=100.0e9),
  # QuadratureIncompressibilityConstraint(weight=0.1e9),
  with_props=True
)
props = Properties(
  # prop_mins=[0.45485, -0.31492, 0.000002, 0.00172], 
  # prop_maxs=[2.98489, -0.04736, 0.450380, 1.86427],
  prop_mins=[0.01],
  prop_maxs=[10.0],
  key=key
)
network = MLP(4, 3, 50, 5, jax.nn.tanh, key)
params = FieldPropertyPair(network, props)
# params = eqx.tree_deserialise_leaves('/projects/sm_pinns/pancax-git/pancax/checkpoint_0990000.eqx', params)

##################
# Training
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=5000)
opt_st = opt.init(params)
# for epoch in range(1000000, 10 * 1000000):
for epoch in range(int(10e6)):
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
