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
mesh_file = find_mesh_file('mesh_quad4.g')
logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'vtk')

##################
# physics setup
##################
times = jnp.linspace(0.0, 1.0, 11)
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
  EssentialBC('nset_3', 1),
]
natural_bcs = [
]
domain = VariationalDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times, q_order=2)

##################
# ML setup
##################
# loss_function = CombineLossFunctions(
#   EnergyAndResidualLoss(residual_weight=250.0),
#   QuadratureIncompressibilityConstraint(weight=100.0)
# )
loss_function = EnergyLoss()
field_network = MLP(physics_kernel.n_dofs + 1, physics_kernel.n_dofs, 50, 5, jax.nn.tanh, key)
# props = FixedProperties([1000.0, 0.3846])
shear = 1.0
props = FixedProperties([1000.0 * shear, shear])
params = FieldPropertyPair(field_network, props)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, clip_gradients=False)
opt_st = opt.init(params)
for epoch in range(1000):
  params, opt_st, loss = opt.step(params, domain, opt_st)
  logger.log_loss(loss, epoch)

##################
# post-processing
##################
pp.init(domain, 'output.vtm', 
  node_variables=[
    'displacement',
    # 'internal_force'
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
