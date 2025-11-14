from pancax import *

##################
# for reproducibility
##################
key = random.PRNGKey(10)
key = random.split(key, 8)

##################
# file management
##################
full_field_data_file = find_data_file('data_full_field.csv')
global_data_file = find_data_file('data_global_data.csv')
mesh_file = find_mesh_file('mesh.g')

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
# domain setup
##################
times = jnp.linspace(0.0, 1.0, len(global_data.outputs))
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
model = NeoHookean(
  bulk_modulus=0.833,
  shear_modulus=BoundedProperty(0.01, 5., key=key)
)
physics = SolidMechanics(model, PlaneStrain())

##################
# ics/bcs
##################
ics = [
]
dirichlet_bc_func = UniaxialTensionLinearRamp(
  final_displacement=jnp.max(field_data.outputs[:, 0]), 
  length=1.0, direction='x', n_dimensions=2
)
# physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)
dirichlet_bcs = [
  DirichletBC('nodeset_2', 0), # left edge fixed in x
  DirichletBC('nodeset_2', 1), # left edge fixed in y
  DirichletBC('nodeset_4', 0), # right edge prescribed in x
  DirichletBC('nodeset_4', 1)  # right edge fixed in y
]
neumann_bcs = [
]

##################
# problem setup
##################
problem = InverseProblem(domain, physics, field_data, global_data, ics, dirichlet_bcs, neumann_bcs)

# print(problem)

##################
# ML setup
##################
params = Parameters(
  problem, key,
  dirichlet_bc_func=dirichlet_bc_func
)#, seperate_networks=True, network_type=ResNet)
print(params)
physics_and_global_loss = EnergyResidualAndReactionLoss(
  residual_weight=250.e9, reaction_weight=250.e9
)
full_field_data_loss = FullFieldDataLoss(weight=10.e9)

def loss_function(params, problem, inputs, outputs):
  loss_1, aux_1 = physics_and_global_loss(params, problem)
  loss_2, aux_2 = full_field_data_loss(params, problem, inputs, outputs)
  # reg = 1.e9 * jnp.sum(jnp.sum(jnp.abs(param)) for param in eqx.filter(params.fields.networks.mlp.layers, eqx.is_array))
  # nn_params = eqx.partition(params.fields, eqx.is_array)
  # reg = 1.e9 * jnp.sum(jnp.sum(jnp.abs(p)) for p in nn_params)
  reg = eqx.filter(params.fields, lambda x: isinstance(x, jax.Array) and x.ndim > 1)
  reg = 1.e7 * sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(reg))
  aux_1.update(aux_2)
  aux_1.update(dict(reg=reg))
  return loss_1 + loss_2 + reg, aux_1

loss_function = UserDefinedLossFunction(loss_function)

opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=500)

##################
# Training
##################
opt, opt_st = opt.init(params)

dataloader = FullFieldDataLoader(problem.field_data)
for epoch in range(100000):
  for inputs, outputs in dataloader.dataloader(1024):
    params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs)

  if epoch % 10 == 0:
    print(epoch)
    print(loss)
    print(params.physics.constitutive_model)
    print(params.physics.constitutive_model.shear_modulus.prop_min)
    print(params.physics.constitutive_model.shear_modulus.prop_max)

    # print(params.physics.constitutive_model.shear_modulus)
