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
    # return u
    return z

def f(xs):
    x, y = xs[0], xs[1]
    return 2. * jnp.pi**2 * jnp.sin(2. * jnp.pi * x) * jnp.sin(2. * jnp.pi * y)

@eqx.filter_jit
def residual(params, domain):
    return vmap(domain.physics.strong_form_residual, in_axes=(None, 0, None))(
        params, domain.coords, 0.0
    )

@eqx.filter_jit
def jacobian(params, domain):
    return vmap(eqx.filter_grad(domain.physics.strong_form_residual), in_axes=(None, 0, None))(
        params, domain.coords, 0.0
    ).fields.beta

@eqx.filter_jit
def residual_bc(params, domain):
    R_bcs = []
    func = lambda p, x, t: domain.physics.field_values(p.fields, x, t)[bc.component]
    for bc in domain.essential_bcs:
        nodes = domain.mesh.nodeSets[bc.nodeSet]
        R_bc = vmap(func, in_axes=(None, 0, None))(params, domain.coords[nodes, :], 0.0)
        R_bcs.append(R_bc)
    return jnp.hstack(R_bcs)

@eqx.filter_jit
def jacobian_bc(params, domain):
    J_bcs = []
    func = lambda p, x, t: domain.physics.field_values(p.fields, x, t)[bc.component]
    func = eqx.filter_grad(func)
    for bc in domain.essential_bcs:
        nodes = domain.mesh.nodeSets[bc.nodeSet]
        J_bc = vmap(func, in_axes=(None, 0, None))(params, domain.coords[nodes, :], 0.0).fields.beta
        J_bcs.append(J_bc)
    return jnp.vstack(J_bcs)


physics_kernel = Poisson(mesh_file, bc_func, f)
essential_bcs = [
    EssentialBC('nset_1', 0)
]
natural_bcs = [
]
times = jnp.linspace(0., 0., 1)
domain = CollocationDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times)

##################
# ML setup
##################
n_inputs = 3
n_neurons = 2000
radius = 3.0
network = ELM(3, 1, n_neurons, key)
props = FixedProperties([])
params = FieldPropertyPair(network, props)
print(params)

HEADER = 'Iteration   |R|               |R|/|R0|           |dB|               cond(J)           rank(J)'
print(HEADER)
R0 = -1.
for n in range(10):
    J = jacobian(params, domain)[domain.dof_manager.unknownIndices, :]
    R = residual(params, domain)[domain.dof_manager.unknownIndices]
    J_bc = jacobian_bc(params, domain)
    R_bc = residual_bc(params, domain)
    J = jnp.vstack((J, J_bc))
    R = jnp.hstack((R, R_bc))
    if n == 0:
        R0 = jnp.linalg.norm(R)
    delta_beta, residuals, rank_J, s = jnp.linalg.lstsq(J, R)
    params = eqx.tree_at(lambda m: m.fields.beta, params, params.fields.beta - delta_beta)
    print('{0:8}    {1:.8e}    {2:.8e}     {3:.8e}     {4:.8e}    {5:8}'.format(
        n, 
        jnp.linalg.norm(R), 
        jnp.linalg.norm(R) / R0, 
        jnp.linalg.norm(delta_beta), 
        jnp.linalg.cond(J), 
        rank_J
    ))

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
