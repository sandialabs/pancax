from pancax import *

##################
# for debugging nans... this will slow things down though.
##################
# jax.config.update("jax_debug_nans", True)

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_hex8.g')
logger = Logger('pinn.log', log_every=100)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
times = jnp.linspace(0.0, 1.0, 5)
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=1.0, direction='y', n_dimensions=3
)
model = NeoHookean()
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
domain = VariationalDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times)
# assert False
##################
# ML setup
##################
# loss_function = EnergyAndResidualLoss(alpha=250.0)
loss_function = ResidualMSELoss(weight=1.0)
# loss_function = EnergyLoss()
field_network = MLP(physics_kernel.n_dofs + 1, physics_kernel.n_dofs, 50, 5, jax.nn.tanh, key)
props = FixedProperties([0.833, 0.3846])
params = FieldPropertyPair(field_network, props)

##################
# train network
##################
# with jax.profiler.trace("jax_profile", create_perfetto_link=True):
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.init(params)
for epoch in range(10000):
  params, opt_st, loss = opt.step(params, domain, opt_st)
  logger.log_loss(loss, epoch)

##################
# post-processing
##################
pp.init(domain, 'output.e', 
  node_variables=[
    'displacement',
  ], 
  element_variables=[
    'element_cauchy_stress'
  ]
)
pp.write_outputs(params, domain)
pp.close()
