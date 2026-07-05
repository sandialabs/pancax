from pancax import *

key = random.PRNGKey(10)
mesh_file = find_mesh_file("mesh_quad4.g")
times = jnp.linspace(0.0, 0.0, 1)
# domain = CollocationDomain(mesh_file, times)
domain = VariationalDomain(mesh_file, times)
ssets = domain.mesh.sideSets
df = distance_function(domain, ssets)

physics = Poisson(lambda x: 2 * jnp.pi**2 * jnp.sin(2. * jnp.pi * x[0]) * jnp.sin(2. * jnp.pi * x[1]))

def bc_func(x, t, z):
    return df(x) * z

ics = [
]
dbcs = [
  DirichletBC(nset_name="nset_1", component=0),
  DirichletBC(nset_name="nset_2", component=0),
  DirichletBC(nset_name="nset_3", component=0),
  DirichletBC(nset_name="nset_4", component=0),
]
nbcs = [
]

problem = ForwardProblem(domain, physics, ics, dbcs, nbcs)
params = Parameters(problem, key, dirichlet_bc_func=bc_func, network_type=ResNet)
print(params)

# # u = jax.vmap(physics.field_values, in_axes=(0, None))(X, 0.0)
# u = jax.vmap(params.fields, in_axes=(0, None))(X, 0.0)
# print(u)
# def loss_function(params, problem, inputs, outputs):
#   field, physics, state = params
#   residuals = jax.vmap(physics.strong_form_residual, in_axes=(None, 0, 0))(
#     field, inputs[:, 0:2], inputs[:, 2:3]
#   )
#   return jnp.square(residuals - outputs).mean(), dict(nothing=0.0)

# loss_function = UserDefinedLossFunction(loss_function)
loss_function = EnergyLoss()
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt, opt_st = opt.init(params)

# dataloader = CollocationDataLoader(problem.domain, num_fields=1)
# for epoch in range(50000):
#   for inputs, outputs in dataloader.dataloader(512):
#     params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs)

#   if epoch % 100 == 0:
#     print(epoch)
#     print(loss)

# pre-train with Adam
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
opt, opt_st = opt.init(params)
for epoch in range(25000):
  params, opt_st, loss = opt.step(params, opt_st, problem)

  if epoch % 100 == 0:
    print(epoch)
    print(loss)
