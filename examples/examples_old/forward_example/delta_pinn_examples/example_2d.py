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
mesh_file = find_mesh_file('ellipse_test.e')
logger = Logger('pinn.log', log_every=100)
history = HistoryWriter('history.csv', log_every=250, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
times = jnp.linspace(0.0, 1.0, 21)
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=2.0, direction='y', n_dimensions=2
)
model = NeoHookean()
# model = Gent()
formulation = PlaneStrain()
physics_kernel = SolidMechanics(mesh_file, essential_bc_func, model, formulation, use_delta_pinn=True)
essential_bcs = [
  EssentialBC('yminus_nodeset', 0),
  EssentialBC('yminus_nodeset', 1),
  EssentialBC('yplus_nodeset', 0),
  EssentialBC('yplus_nodeset', 1)
]
natural_bcs = [
]
domain = DeltaPINNDomain(
  physics_kernel, essential_bcs, natural_bcs, mesh_file, times,
  n_eigen_values=20
)

##################
# ML setup
##################
# loss_function = ResidualMSELoss(use_delta_pinn=True)
loss_function = EnergyLoss()
field_network = MLP(20 + 1, 2, 20, 3, jax.nn.tanh, key)
props = FixedProperties([0.833, 0.3846])
params = FieldPropertyPair(field_network, props)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.init(params)
for epoch in range(25000):
  params, opt_st, loss = opt.step(params, domain, opt_st)
  logger.log_loss(loss, epoch)
  history.write_history(loss, epoch)

##################
# post-processing
##################
pp.init(domain, 'output.e', 
  node_variables=[
    'displacement'
  ], 
  element_variables=[
    # 'element_cauchy_stress',
    # 'element_displacement_gradient',
    # 'element_deformation_gradient',
    'element_invariants',
    # 'element_pk1_stress'
  ]
)
# pp.write_outputs(params, domain)
def field_values(field_network, physics, x, t, v):
    # v_temp = (v - jnp.min(v, axis=0)) / (jnp.max(v, axis=0) - jnp.min(v, axis=0))
    # inputs = jnp.hstack((v_temp, t))
    inputs = jnp.hstack((v, t))
    z = field_network(inputs)
    u = physics.bc_func(x, t, z)
    return u

for n, time in enumerate(domain.times):
  pp.exo.put_time(n + 1, time)
  us = jax.vmap(field_values, in_axes=(None, None, 0, None, 0))(
     params.fields, domain.physics, domain.coords, time, domain.eigen_modes
  )
  us = onp.asarray(us)
  pp.exo.put_node_variable_values('displ_x', n + 1, us[:, 0])
  pp.exo.put_node_variable_values('displ_y', n + 1, us[:, 1])


pp.close()
