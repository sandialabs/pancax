from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4.g')
logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 1.0, 11)
domain = VariationalDomain(mesh_file, times, q_order=2)

##################
# physics setup
##################
dirichlet_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=1.0, direction='y', n_dimensions=2
)
model = NeoHookean(
  bulk_modulus=1000.0,
  shear_modulus=1.,
)
physics = SolidMechanics(model, PlaneStrain())
physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)
ics = [
]
dirichlet_bcs = [
  DirichletBC('nset_1', 0),
  DirichletBC('nset_1', 1),
  DirichletBC('nset_3', 0),
  DirichletBC('nset_3', 1),
]
neumann_bcs = [
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs)

##################
# ML setup
##################
loss_function = EnergyLoss()
params = Parameters(problem, key, seperate_networks=True)
print(params)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, clip_gradients=False)
opt_st = opt.init(params)
for epoch in range(100000):
  params, opt_st, loss = opt.step(params, problem, opt_st)
  logger.log_loss(loss, epoch)

##################
# post-processing
##################
pp.init(problem, 'output.e', 
  node_variables=[
    'field_values'
    # 'displacement',
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
pp.write_outputs(params, problem)
pp.close()
