from pancax import *
import jax.tree_util as jtu

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
times = jnp.linspace(0.0, 1.0, 11)
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=0.5, length=1.0, direction='y', n_dimensions=3
)
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, NeoHookeanFixedBulkModulus(100.0), ThreeDimensional(), use_delta_pinn=False)
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
domain = InverseDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times, field_data, global_data)

##################
# ML setup
##################
field_network = MLP(3 + 1, 3, 100, 3, jax.nn.tanh, key)
props = Properties = Properties(
  prop_mins=[0.01],
  prop_maxs=[10.0],
  key=key,
  activation_func=jax.nn.sigmoid
)
params = FieldPropertyPair(field_network, props)
filter_spec_1 = params.freeze_props_filter()
filter_spec_2 = params.freeze_fields_filter()

##################
# Restart
##################
# params = eqx.tree_deserialise_leaves('checkpoint_1250000.eqx', params)

##################
# train network
##################
loss_function_1 = EnergyLoss()
loss_function_2 = CombineLossFunctions(
  EnergyResidualAndReactionLoss(energy_weight=1.0, residual_weight=0.001e9, reaction_weight=10.0e9),
  FullFieldDataLoss(weight=1.0e9),
  with_props=True 
)
opt_1 = Adam(loss_function_1, learning_rate=1.0e-3, has_aux=True, filter_spec=filter_spec_1)
opt_2 = Adam(loss_function_2, learning_rate=1.0e-3, has_aux=True, transition_steps=2500, filter_spec=filter_spec_2)
trainer_1 = Trainer(
  domain, opt_1, 
  output_node_variables=['displacement'], 
  log_every=2500, 
  output_every=100000
)
trainer_2 = Trainer(
  domain, opt_2, 
  output_node_variables=['displacement'], 
  log_file='props_log.log',
  log_every=10,
  output_every=1000000000
)
opt_st_2 = trainer_2.init(params)

# pre-train on initial prop guesses
print('Pre-training')
params = trainer_1.train(params, 25000)
for n in range(1000):
  print(f'Iteration = {n}')
  params = trainer_1.train(params, 5000)
  for m in range(50):
    params, opt_st_2 = trainer_2.step(params, opt_st_2)

# params = trainer_1.train(params, 10000)
# params = trainer_2.train(params, 1000000)