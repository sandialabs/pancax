from pancax import *

##################
# for debugging nans... this will slow things down though.
##################
# jax.config.update("jax_debug_nans", True)

##################
# for reproducibility
##################
n_pinns = 8
key = jax.random.PRNGKey(0)
key = jax.random.split(key, 8)

##################
# file management
##################
full_field_data_file = find_data_file('data_full_field.csv')
global_data_file = find_data_file('global_data.csv')
mesh_file = find_mesh_file('mesh.g')
logger = EnsembleLogger('pinn', n_pinns, log_every=250)
history = EnsembleHistoryWriter('history', n_pinns, log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 'z', 't'], ['u_x', 'u_y', 'u_z'])
# the 6 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 't', 'u_x', 'f_x',
  mesh_file, 6, 'x', # these inputs specify where to measure reactions
  n_time_steps=11, # the number of time steps for inverse problems is specified here
  plotting=True
)

##################
# physics setup
##################
times = global_data.times
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 0]), 
  length=1.0, direction='x', n_dimensions=3
)
model = Swanson4FixedBulkModulus(bulk_modulus=10.0, cutoff_strain=0.01)
formulation = ThreeDimensional()
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, model, formulation)
essential_bcs = [
  EssentialBC('nset_4', 0),
  EssentialBC('nset_4', 1),
  EssentialBC('nset_4', 2),
  EssentialBC('nset_6', 0),
  EssentialBC('nset_6', 1),
  EssentialBC('nset_6', 2)
]
natural_bcs = [
]
domain = InverseDomain(
  physics_kernel, essential_bcs, natural_bcs, mesh_file, times,
  field_data, global_data#, q_order=1
)

##################
# ML setup
##################
physics_loss = EnergyResidualAndReactionLoss(alpha=250.0e9, beta=250.0e9)
full_field_loss = FullFieldDataLoss(essential_bc_func, weight=10.0e9)

def loss_function(params, domain):
  _, props = params
  p_loss, p_aux = physics_loss(params, domain, times)
  u_loss, u_aux = full_field_loss(params, domain)
  p_aux.update(u_aux)
  p_aux.update({'props': props()}) # predict props again just for logging
  return p_loss + u_loss, p_aux

@eqx.filter_vmap
def make_params(key):
  field_network = MLP(4, 3, 20, 3, jax.nn.tanh, key)
  props = Properties(
    prop_mins=[0.45485, -0.31492, 0.000002, 0.00172], 
    prop_maxs=[2.98489, -0.04736, 0.450380, 1.86427],
    key=key
  )  
  params = FieldPropertyPair(field_network, props)
  return params

params = make_params(key)

##################
# Training
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.ensemble_init(params)

for epoch in range(100000):
  params, opt_st, loss = opt.ensemble_step(params, domain, opt_st)
  logger.log_loss(loss, epoch)
  history.write_history(loss, epoch)

  # if epoch % 10000 == 0:
  #   def vmap_func(p):
  #     p.serialise('checkpoint', epoch)

  #   eqx.filter