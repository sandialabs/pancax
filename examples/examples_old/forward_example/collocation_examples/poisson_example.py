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
mesh_file = find_mesh_file('mesh_1x.g')
logger = Logger('pinn.log', log_every=100)
history = HistoryWriter('history.csv', log_every=100, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
def bc_func(xs, t, z):
    x, y = xs[0], xs[1]
    u = z
    # u = u.at[0].set(
    #     x * y * (1. - x) * (1. - y) * z[0]
    # )
    # u = u.at[0].set(
    #     x * y * (1. - x) * (1. - y) * z[0] + x
    # )
    return u

def f(xs):
    x, y = xs[0], xs[1]
    return 2. * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

physics_kernel = Poisson(mesh_file, bc_func, f)
essential_bcs = [
    # EssentialBC('nodeset_1', 0),
    # EssentialBC('nodeset_2', 0),
    # EssentialBC('nodeset_3', 0),
    # EssentialBC('nodeset_4', 0)
]
natural_bcs = [
    NaturalBC('sideset_1'),
    NaturalBC('sideset_2'),
    NaturalBC('sideset_3'),
    NaturalBC('sideset_4')
]
times = jnp.linspace(0., 0., 1)
domain = CollocationDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times, q_order=4)

##################
# ML setup
##################
# loss_function = StrongFormResidualLoss()
loss_function = CombineLossFunctions(
    NeumannBCLoss(weight=1),
    StrongFormResidualLoss()
)
field_network = MLP(3, 1, 20, 3, jax.nn.tanh, key)
props = FixedProperties([])
params = FieldPropertyPair(field_network, props)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True)
opt_st = opt.init(params)
for epoch in range(50000):
    params, opt_st, loss = opt.step(params, domain, opt_st)
    logger.log_loss(loss, epoch)
    history.write_history(loss, epoch)

##################
# post-processing
##################
pp.init(domain, 'output.e',
    node_variables=[
        'field_values'
    ]        
)
pp.write_outputs(params, domain)
pp.close()
