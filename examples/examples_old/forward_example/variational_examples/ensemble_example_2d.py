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
mesh_file = find_mesh_file('mesh_quad4.g')
logger = EnsembleLogger('pinn', n_pinns, log_every=100)
history = EnsembleHistoryWriter('history', n_pinns, log_every=1, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
times = jnp.linspace(0.0, 1.0, 2)
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=1.0, direction='y', n_dimensions=2
)
model = NeoHookean()
formulation = PlaneStrain()
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, model, formulation)
essential_bcs = [
  EssentialBC('nset_1', 0),
  EssentialBC('nset_1', 1),
  EssentialBC('nset_3', 0),
  EssentialBC('nset_3', 1)
]
natural_bcs = [
]
domain = ForwardDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times)

##################
# ML setup
##################
loss_function = EnergyAndResidualLoss(alpha=250.0)

@eqx.filter_vmap
def make_params(key):
  field_network = MLP(3, 2, 50, 5, jax.nn.tanh, key)
  props = FixedProperties([0.833, 0.3846])
  params = FieldPropertyPair(field_network, props)
  return params

params = make_params(key)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.ensemble_init(params)

for epoch in range(1000):
  params, opt_st, loss = opt.ensemble_step(params, domain, opt_st)
  logger.log_loss(loss, epoch)
  history.write_history(loss, epoch)
