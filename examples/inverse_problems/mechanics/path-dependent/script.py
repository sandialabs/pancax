from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
full_field_data_file = find_data_file('full_field_data.csv')
global_data_file = find_data_file('global_data.csv')
mesh_file = find_mesh_file('mesh_quad4.g')
# logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history_pd.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# data setup
##################
field_data = FullFieldData(full_field_data_file, ['x', 'y', 't'], ['u_x', 'u_y'])
# the 4 below is for the nodeset id
global_data = GlobalData(
  global_data_file, 'times', 'disps', 'forces',
  mesh_file, 1, 'y', # these inputs specify where to measure reactions
  n_time_steps=21, # the number of time steps for inverse problems is specified here
  plotting=True,
  interpolate=False
)
times = global_data.times

##################
# domain setup
##################
domain = VariationalDomain(mesh_file, times, q_order=2)

##################
# physics setup
##################
model = NeoHookean(
  bulk_modulus=10.0,
  # bulk_modulus=BoundedProperty(0.01, 5., key=key),
  shear_modulus=BoundedProperty(0.01, 5., key=key)
)
model = SimpleFeFv(
    # NeoHookean(bulk_modulus=10.0, shear_modulus=1.0),
    model,
    PronySeries(moduli=[1.0], relaxation_times=[10.0]),
    WLF(C1=17.44, C2=51.6, theta_ref=60.0),
)
physics = SolidMechanics(model, PlaneStrain())

##################
# ics/bcs
##################
ics = [
]
def dirichlet_bc_func(xs, t, nn):
    length = 1.
    final_displacement = 1.
    # x, y, z = xs[0], xs[1], xs[2]
    y = xs[1]
    u_out = nn
    u_out = u_out.at[0].set(
        y * (y - length) * t * nn[0] / length**2
    )

    u_out = jax.lax.cond(
        t > 1.,
        lambda u: u.at[1].set(
            y * final_displacement / length
            + y * (y - length) * t * nn[1] / length**2
        ),
        lambda u: u.at[1].set(
            y * t * final_displacement / length
            + y * (y - length) * t * nn[1] / length**2
        ),
        u_out
    )
    # u_out = u_out.at[2].set(
    #     y * (y - length) * t * nn[2] / length**2
    # )
    return u_out

# model = NeoHookean(bulk_modulus=10., shear_modulus=0.855)
physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)
dirichlet_bcs = [
  DirichletBC('nset_1', 0), # left edge fixed in x
  DirichletBC('nset_1', 1), # left edge fixed in y
  DirichletBC('nset_3', 0), # right edge prescribed in x
  DirichletBC('nset_3', 1)  # right edge fixed in y
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
params = Parameters(problem, key, seperate_networks=False, network_type=MLP)
physics_and_global_loss = PathDependentEnergyResidualAndReactionLoss(
  residual_weight=250.e9, reaction_weight=250.e9
)
full_field_data_loss = FullFieldDataLoss(weight=10.e9)

def loss_function(params, problem, inputs, outputs):
  loss_1, aux_1 = physics_and_global_loss(params, problem)
  loss_2, aux_2 = full_field_data_loss(params, problem, inputs, outputs)
  aux_1.update(aux_2)
  return loss_1 + loss_2, aux_1

loss_function = UserDefinedLossFunction(loss_function)
# loss_function = EnergyLoss()
opt = Adam(
  loss_function, 
  learning_rate=1.0e-3, 
  has_aux=True, 
  transition_steps=50000,
  jit=True
)

##################
# Training
##################
print(params)
opt_st = opt.init(params)


# testing stuff
dataloader = FullFieldDataLoader(problem.field_data)

for epoch in range(1000000):
  for inputs, outputs in dataloader.dataloader(8 * 1024):
    params, opt_st, loss = opt.step(params, problem, opt_st, inputs, outputs)
    # params, opt_st, loss = opt.step(params, problem, opt_st)


  history.write_data("epoch", epoch)
  history.write_loss(loss)
  history.write_data("shear_modulus", params.physics.constitutive_model.eq_model.shear_modulus)

  print(epoch)
  print(loss)

  if epoch % 1000 == 0:
    history.to_csv()
