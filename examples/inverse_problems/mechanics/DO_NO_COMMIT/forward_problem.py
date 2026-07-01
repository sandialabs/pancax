from pancax import *


##################
# for reproducibility
##################
key = random.key(100)

##################
# file management
##################
mesh_file = './mesh/mesh.g'
logger = Logger('pinn.log', log_every=250)
history = HistoryWriter('history.csv', log_every=250, write_every=250)
pp = PostProcessor(mesh_file)

##################
# domain setup
##################
times = jnp.linspace(0., 1., 11)
domain = VariationalDomain(mesh_file, times)

##################
# physics setup
##################
model = NeoHookean(
  bulk_modulus=980.0,
  shear_modulus=2.
)
physics = SolidMechanics(model, ThreeDimensional())

##################
# ics/bcs setup
##################
ics = [
]
essential_bcs = [
  EssentialBC('nset_bottom', 0),
  EssentialBC('nset_bottom', 1),
  EssentialBC('nset_bottom', 2),
  EssentialBC('nset_top', 0),
  EssentialBC('nset_top', 1, lambda x, t: -25.4 * t),
  EssentialBC('nset_top', 2) 
]
natural_bcs =[
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, essential_bcs, natural_bcs)
print(problem)

##################
# ML setup
##################
field_network = MLP(3 + 1, 3, 100, 3, jax.nn.tanh, key)
params = FieldPhysicsPair(field_network, problem.physics)

##################
# train network
##################
# loss_function = EnergyLoss()
loss_function = CombineLossFunctions(
  EnergyLoss(),
  DirichletBCLoss(weight=1.e4)
)

opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)

opt_st = opt.init(params)

for epoch in range(25000):
  params, opt_st, loss = opt.step(params, problem, opt_st)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
    print(params.physics.x_mins)
    print(params.physics.x_maxs)


  if epoch % 1000 == 0:
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
