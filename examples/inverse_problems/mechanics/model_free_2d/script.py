from pancax import *

##################
# for reproducibility
##################
n_ensemble = 4
is_ensemble = False
key = random.PRNGKey(10)
key_f, key_m = jax.random.split(key)
# key = random.split(key, 8)
if is_ensemble:
    key_f = jax.random.split(key_f, n_ensemble)
    key_m = jax.random.split(key_m, n_ensemble)

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
# model = NeoHookean(
#   bulk_modulus=0.833,
#   shear_modulus=BoundedProperty(0.01, 5., key=key_m)
# )
model = InputPolyConvexPotential(bulk_modulus=0.833, key=key_m, use_l0_regularization=True)
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
  problem, key_f,
  dirichlet_bc_func=dirichlet_bc_func
)#, seperate_networks=True, network_type=ResNet)
print(params)
physics_and_global_loss = EnergyResidualAndReactionLoss(
  residual_weight=250.e9, reaction_weight=250.e9
)
full_field_data_loss = FullFieldDataLoss(weight=100.e9)

gate_key = random.PRNGKey(100)

def loss_function(params, problem, inputs, outputs, gate_key):
#   jax.debug.print("Gate key = {x}", x=gate_key)
#   L = 5
#   loss_1 = 0.0
#   for n in range(L): 
  loss_1, aux_1 = physics_and_global_loss(params, problem, gate_key)
#   loss_1 = loss_1 + loss_temp

#   loss_1 = loss_1 / L
  loss_2, aux_2 = full_field_data_loss(params, problem, inputs, outputs)
  # reg = 1.e9 * jnp.sum(jnp.sum(jnp.abs(param)) for param in eqx.filter(params.fields.networks.mlp.layers, eqx.is_array))
  # nn_params = eqx.partition(params.fields, eqx.is_array)
  # reg = 1.e9 * jnp.sum(jnp.sum(jnp.abs(p)) for p in nn_params)
  reg_f = eqx.filter(params.fields, lambda x: isinstance(x, jax.Array) and x.ndim > 1)
#   reg = 1.e7 * sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(reg))
  reg_f = 0 * sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(reg_f))

#   reg_m = 1 * params.physics.constitutive_model.l0_regularization_term(gate_key, 1e-4)
#   reg_m = 0
  reg_m = eqx.filter(params.physics.constitutive_model, lambda x: isinstance(x, jax.Array) and x.ndim > 1)
  reg_m = 0 * sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(reg_m))
  aux_1.update(aux_2)
  aux_1.update(dict(reg_f=reg_f, reg_m=reg_m))
  return loss_1 + loss_2 + reg_f + reg_m, aux_1

loss_function = UserDefinedLossFunction(loss_function)

opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, transition_steps=500)

##################
# Training
##################
opt, opt_st = opt.init(params)

dataloader = FullFieldDataLoader(problem.field_data)
gate_key = random.PRNGKey(100)
for epoch in range(100000):
  for inputs, outputs in dataloader.dataloader(1024):
    gate_key, subkey = jax.random.split(gate_key)
    params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs, subkey)

  if epoch % 10 == 0:
    print("Epoch")
    print(epoch)
    print("Loss")
    print(loss[0])
    print("Energy")
    print(loss[1]["energy"])
    print("Residual")
    print(loss[1]["residual"])
    print("Field data loss")
    print(loss[1]["field_data_loss"])
    print("Global data loss")
    print(loss[1]["global_data_loss"])
    print("Field Regularization")
    print(loss[1]["reg_f"])
    print("Model Regularization")
    print(loss[1]["reg_m"])
    # print("Shear modulus")
    print("Constitutive model parameters")
    print(params.physics.constitutive_model.num_effective_parameters(gate_key))
    # print(params.physics.constitutive_model.network.x1_xx_pos.weight)
    # print(params.physics.constitutive_model.shear_modulus)
    # print(params.physics.constitutive_model)
    # print(params.physics.constitutive_model.shear_modulus.prop_min)
    # print(params.physics.constitutive_model.shear_modulus.prop_max)

    # print(params.physics.constitutive_model.shear_modulus)
