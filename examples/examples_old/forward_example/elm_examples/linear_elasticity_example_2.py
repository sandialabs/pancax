from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('problem_4_1x.g')
logger = Logger('pinn.log', log_every=100)
history = HistoryWriter('history.csv', log_every=100, write_every=1000)
pp = PostProcessor(mesh_file)

##################
# physics setup
##################
lambda_ = 1.0
mu = 0.5

def bc_func(xs, t, z):
    x, y = xs[0], xs[1]
    u = z
    u = u.at[0].set(
        x * u[0]
    )
    u = u.at[1].set(
        y * u[1]
    )
    return u
    # return z

def f(x, t):
    return jnp.array([0., 0.])

@eqx.filter_jit
def residual(params, domain):
    return vmap(domain.physics.strong_form_residual, in_axes=(None, 0, None))(
        params, domain.coords, 0.0
    ).flatten()[domain.dof_manager.unknownIndices]

@eqx.filter_jit
def jacobian(params, domain):
    grads = vmap(eqx.filter_jacrev(domain.physics.strong_form_residual), in_axes=(None, 0, None))(
        params, domain.coords, 0.0
    ).fields.beta
    return grads.reshape((grads.shape[0] * grads.shape[1], grads.shape[2]))[domain.dof_manager.unknownIndices, :]

@eqx.filter_jit
def residual_bc(params, domain):
    R_bcs = []
    # func = lambda p, x, t: domain.physics.field_values(p.fields, x, t)[bc.component]
    # for bc in domain.essential_bcs:
    #     nodes = domain.mesh.nodeSets[bc.nodeSet]
    #     R_bc = vmap(func, in_axes=(None, 0, None))(params, domain.coords[nodes, :], 0.0)
    #     R_bcs.append(R_bc)

    bc_comps = [0, 0, 1, 1, 1, 0]
    for i, bc in enumerate(domain.natural_bcs):
        func = lambda p, x, t, n: domain.physics.strong_form_neumann_bc(p, x, t, n)[bc_comps[i]]
        xs, ns = domain.neumann_xs[i], domain.neumann_ns[i]
        bc_pred = vmap(func, in_axes=(None, 0, None, 0))(params, xs, 0.0, ns)
        bc_exp = vmap(lambda x, t, n: bc.function(x, t, n)[bc_comps[i]], in_axes=(0, None, 0))(xs, 0.0, ns)
        R_bcs.append(bc_pred - bc_exp)
    
    return jnp.hstack(R_bcs)

@eqx.filter_jit
def jacobian_bc(params, domain):
    J_bcs = []
    # func = lambda p, x, t: domain.physics.field_values(p.fields, x, t)[bc.component]
    # func = eqx.filter_grad(func)
    # for bc in domain.essential_bcs:
    #     nodes = domain.mesh.nodeSets[bc.nodeSet]
    #     J_bc = vmap(func, in_axes=(None, 0, None))(params, domain.coords[nodes, :], 0.0).fields.beta
    #     J_bcs.append(J_bc)

    bc_comps = [0, 0, 1, 1, 1, 0]
    for i, bc in enumerate(domain.natural_bcs):
        func = lambda p, x, t, n: domain.physics.strong_form_neumann_bc(p, x, t, n)[bc_comps[i]]
        func = eqx.filter_grad(func)
        xs, ns = domain.neumann_xs[i], domain.neumann_ns[i]
        J_bc = vmap(func, in_axes=(None, 0, None, 0))(params, xs, 0.0, ns).fields.beta
        J_bcs.append(J_bc)
    return jnp.vstack(J_bcs)


physics_kernel = LinearElasticity2D(mesh_file, bc_func, f)
essential_bcs = [
    EssentialBC('nodeset_1', 1),
    EssentialBC('nodeset_2', 0),
    #
    EssentialBC('nodeset_3', 0),
    EssentialBC('nodeset_3', 1),
    EssentialBC('nodeset_4', 0),
    EssentialBC('nodeset_4', 1),
    #
    EssentialBC('nodeset_1', 0),
    EssentialBC('nodeset_2', 1)
]
natural_bcs = [
    # NaturalBC('sideset_3', lambda x, t, n: jnp.zeros(2)),
    NaturalBC('sideset_3', lambda x, t, n: n),
    NaturalBC('sideset_4', lambda x, t, n: 3. * n),
    # NaturalBC('sideset_3', lambda x, t, n: jnp.zeros(2)),
    NaturalBC('sideset_3', lambda x, t, n: n),
    NaturalBC('sideset_4', lambda x, t, n: 3. * n),
    #
    NaturalBC('sideset_1', lambda x, t, n: jnp.zeros(2)),
    NaturalBC('sideset_2', lambda x, t, n: jnp.zeros(2))
]
times = jnp.linspace(0., 0., 1)
domain = CollocationDomain(physics_kernel, essential_bcs, natural_bcs, mesh_file, times, q_order=4)

##################
# ML setup
##################
n_inputs = 3
n_neurons = 8000
radius = 3.0
network = ELM(3, 2, n_neurons, key)
props = FixedProperties([lambda_, mu])

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
    'field_values',
    'field_gradients',
    'linear_strain',
    'cauchy_stress'
  ], 
  element_variables=[
  ]
)
pp.write_outputs(params, domain)
pp.close()
