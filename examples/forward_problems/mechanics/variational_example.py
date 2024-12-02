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
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
model = Gent(
  bulk_modulus=0.833,
  shear_modulus=0.3846,
  Jm_parameter=3.
)
physics = SolidMechanics(model, PlaneStrain())

##################
# ics/bcs
##################
ics = [
]
essential_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=1.0, direction='y', n_dimensions=2
)
physics = physics.update_dirichlet_bc_func(essential_bc_func)
essential_bcs = [
  EssentialBC('nset_1', 0),
  EssentialBC('nset_1', 1),
  EssentialBC('nset_3', 0),
  EssentialBC('nset_3', 1),
]
natural_bcs = [
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, essential_bcs, natural_bcs)

##################
# ML setup
##################
n_dims = domain.coords.shape[1]
field = MLP(n_dims + 1, physics.n_dofs, 50, 3, jax.nn.tanh, key)
params = FieldPropertyPair(field, problem.physics)

loss_function = EnergyLoss()
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt_st = opt.init(params)

for epoch in range(10000):
  params, opt_st, loss = opt.step(params, problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    # print(params.properties.constitutive_model.shear_modulus())

##################
# post-processing
##################
pp.init(problem, 'output.e', 
  node_variables=[
    # 'displacement'
    'field_values'
  ], 
  element_variables=[]
)
pp.write_outputs(params, problem)
pp.close()
