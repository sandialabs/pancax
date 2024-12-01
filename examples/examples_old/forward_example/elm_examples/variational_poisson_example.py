from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('problem_1.g')
logger = Logger('pinn.log', log_every=100)
history = HistoryWriter('history.csv', log_every=100, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
def bc_func(xs, t, z):
    x, y = xs[0], xs[1]
    u = x * y * (1. - x) * (1. - y) * z
    return u

def f(xs):
    x, y = xs[0], xs[1]
    return 2. * jnp.pi**2 * jnp.sin(2. * jnp.pi * x) * jnp.sin(2. * jnp.pi * y)

def residual(basis, betas, props, domain):
    fields = ELM2(basis, betas, 1)
    us = domain.field_values(fields, 0.0)
    return internal_force(domain, us, props()).flatten()[domain.dof_manager.unknownIndices]

jacobian = eqx.filter_jit(jacfwd(residual, argnums=1))
residual = eqx.filter_jit(residual)

physics_kernel = Poisson(mesh_file, bc_func, f)
essential_bcs = [
    EssentialBC('nset_1', 0)
]
natural_bcs = [
]
times = jnp.linspace(0., 0., 1)
domain = VariationalDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times)

##################
# ML setup
##################
n_inputs = 3
n_neurons = 1200
radius = 3.0
network = ELM(3, 1, n_neurons, key)
basis, beta = network.layer, network.beta
props = FixedProperties([])
params = FieldPropertyPair(network, props)
print(params)

HEADER = 'Iteration   |R|               |R|/|R0|           |dB|               cond(J)           rank(J)'
print(HEADER)
R0 = -1.
for n in range(10):
    J = jacobian(basis, beta, props, domain)
    J = jacobian(basis, beta, props, domain)[domain.dof_manager.unknownIndices, :]
    R = residual(basis, beta, props, domain)[domain.dof_manager.unknownIndices]

    if n == 0:
        R0 = jnp.linalg.norm(R)

    delta_beta, residuals, rank_J, s = jnp.linalg.lstsq(J, R)
    beta = beta - delta_beta
    print('{0:8}    {1:.8e}    {2:.8e}     {3:.8e}     {4:.8e}    {5:8}'.format(
        n, 
        jnp.linalg.norm(R), 
        jnp.linalg.norm(R) / R0, 
        jnp.linalg.norm(delta_beta), 
        jnp.linalg.cond(J), 
        rank_J
    ))

# setup another ELM2
params = FieldPropertyPair(ELM2(basis, beta, 1), props)

# post-processing
pp.init(domain, 'output.e', 
  node_variables=[
    'field_values'
  ], 
  element_variables=[
  ]
)
pp.write_outputs(params, domain)
pp.close()
