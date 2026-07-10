
import optax
from pancax import *
from pancax.networks.base import create_ensemble
from pancax.networks.parameters import Parameters_v2

key = random.PRNGKey(10)
# key = random.split(key, 8)
mesh_file = find_mesh_file("mesh_quad4.g")
times = jnp.linspace(0.0, 0.0, 1)
# domain = CollocationDomain(mesh_file, times)
domain = VariationalDomain(mesh_file, times)

# physics = Poisson(lambda x: 2 * jnp.pi**2 * jnp.sin(2. * jnp.pi * x[0]) * jnp.sin(2. * jnp.pi * x[1]))
physics = Poisson(lambda x: 1.0)
network = ResNet(3, 1, 50, 5, jax.nn.tanh, key)

# physics = create_ensemble(Poisson, False, key, lambda x: 2 * jnp.pi**2 * jnp.sin(2. * jnp.pi * x[0]) * jnp.sin(2. * jnp.pi * x[1]))
# network = create_ensemble(MLP, True, key, 3, 1, 50, 5, jax.nn.tanh)

ics = [
]
dbcs = [
  DirichletBC(nset_name="nset_3", sset_name="sset_3", component=0, function=lambda x, t: 1.0),
  DirichletBC(nset_name="nset_4", sset_name="sset_4", component=0, function=lambda x, t: 1.0),
  # DirichletBC(nset_name="nset_3", sset_name="sset_3", component=0),
  # DirichletBC(nset_name="nset_4", sset_name="sset_4", component=0),
]
nbcs = [
]

problem = ForwardProblem(domain, physics, ics, dbcs, nbcs)
params = Parameters_v2(key, problem, network)
# params = Parameters(problem, key, dirichlet_bc_func=bc_func, network_type=ResNet)

print(params)
# print(tf)
# print(tf(problem.coords, 0.0))
# out = jax.vmap(tf, in_axes=(0, None))(problem.coords, 0.0)
# print(out)
# params = Parameters(problem, key, dirichlet_bc_func=bc_func, network_type=ResNet)
# print(params)

# # # u = jax.vmap(physics.field_values, in_axes=(0, None))(X, 0.0)
# # u = jax.vmap(params.fields, in_axes=(0, None))(X, 0.0)
# # print(u)
# # def loss_function(params, problem, inputs, outputs):
# #   field, physics, state = params
# #   residuals = jax.vmap(physics.strong_form_residual, in_axes=(None, 0, 0))(
# #     field, inputs[:, 0:2], inputs[:, 2:3]
# #   )
# #   return jnp.square(residuals - outputs).mean(), dict(nothing=0.0)


# # loss_function = UserDefinedLossFunction(loss_function)

loss_function = EnergyLoss()
opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
# opt = LBFGS(loss_function, has_aux=True)
opt_st = opt.init(params)


# filter_spec = params.freeze_physics_normalization_filter()
# scheduler = optax.exponential_decay(
#     init_value=1e-3,
#     transition_steps=500,
#     decay_rate=0.99
# )
# opt = optax.chain(
#     optax.scale_by_adam(),
#     optax.scale_by_schedule(scheduler),
#     optax.scale(-1.0),
# )
# # opt_st = opt.init(eqx.filter(eqx.filter(params, filter_spec), eqx.is_inexact_array))
# opt_st = opt.init(eqx.filter(eqx.filter(params, filter_spec), eqx.is_inexact_array))

# @eqx.filter_jit
# def train_step(params, opt_st, problem):
#     diff_params, static_params = eqx.partition(params, filter_spec)
#     losses, grads = eqx.filter_vmap(
#         eqx.filter_value_and_grad(loss_function.filtered_loss, has_aux=True),
#         in_axes=(eqx.if_array(0), eqx.if_array(0), None),
#     )(diff_params, static_params, problem)

#     grads = eqx.filter(
#         eqx.filter(grads, filter_spec),
#         eqx.is_inexact_array,
#     )

#     updates, opt_st = opt.update(grads, opt_st)
#     params = eqx.apply_updates(params, updates)

#     return params, opt_st, losses

# # print(opt_st)
# assert False
# def make_step_method(filter_spec):
#   def step(params, opt_st, *args):
#     diff_params, static_params = eqx.partition(params, filter_spec)
#     loss, grads = loss_and_grads(diff_params, static_params, *args)
#     updates, opt_st = opt.update(grads, opt_st)
#     params = eqx.apply_updates(params, updates)
#     return params, opt_st, loss

#   return step

# def loss_func(diff_params, static_params, *args):
#   loss = eqx.f

# step = eqx.filter_jit(make_step_method(filter_spec))

# # dataloader = CollocationDataLoader(problem.domain, num_fields=1)
# # for epoch in range(50000):
# #   for inputs, outputs in dataloader.dataloader(512):
# #     params, opt_st, loss = opt.step(params, opt_st, problem, inputs, outputs)

# #   if epoch % 100 == 0:
# #     print(epoch)
# #     print(loss)

# # pre-train with Adam
# opt = Adam(loss_function, learning_rate=1e-3, has_aux=True)
# opt, opt_st = opt.init(params)
for epoch in range(5000):
  # params, opt_st, loss = opt.step(params, opt_st, problem)
  params, opt_st, loss = opt.step(params, opt_st, problem)
  # params, opt_st, loss = train_step(params, opt_st, problem)
  if epoch % 100 == 0:
    print(epoch)
    print(loss)

pp = PostProcessor(mesh_file, 'exodus')
pp.init(params, problem, 'output.e',
  node_variables=['field_values']        
)
pp.write_outputs(params, problem)
pp.close()