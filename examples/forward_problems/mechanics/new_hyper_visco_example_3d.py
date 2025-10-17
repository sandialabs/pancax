from pancax import *
from jax import vmap
# import jax
import netCDF4 as nc

##################
# for reproducibility
##################
key = random.PRNGKey(10)

##################
# file management
##################
mesh_file = find_mesh_file("2holes.g")
output_file = "output-energy.e"
# mesh_file = find_mesh_file("mesh_quad4.g")
pp = PostProcessor(mesh_file, "exodus")

##################
# domain setup
##################
times_1 = jnp.linspace(0., 1., 11)
times_2 = jnp.linspace(1., 2., 11)
times = jnp.hstack((times_1, times_2[1:]))
domain = VariationalDomain(mesh_file, times, q_order=2)

sample_length = 1.
applied_displ = 1.
t_switch = times_1[-1]

def dirichlet_bc_func(xs, t, nn):
    length = sample_length
    final_displacement = applied_displ
    
    # x, y, z = xs[0], xs[1], xs[2]
    y = xs[1]
    u_out = nn
    u_out = u_out.at[0].set(
        y * (y - length) * t * nn[0] / length**2
    )

    u_out = jax.lax.cond(
        t > t_switch,
        lambda u: u.at[1].set(
            y * final_displacement / length
            + y * (y - length) * t * nn[1] / length**2
        ),
        lambda u: u.at[1].set(
            y * (t / t_switch) * final_displacement / length
            + y * (y - length) * t * nn[1] / length**2
        ),
        u_out
    )
    u_out = u_out.at[2].set(
        y * (y - length) * t * nn[2] / length**2
    )
    return u_out

model = SimpleFeFv(
    NeoHookean(bulk_modulus=10.0, shear_modulus=0.855),
    PronySeries(moduli=[1.0], relaxation_times=[0.25]),
    WLF(C1=17.44, C2=51.6, theta_ref=60.0),
)
physics = SolidMechanics(model, ThreeDimensional())
physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)

ics = []
dirichlet_bcs = [
    DirichletBC("nodeset_3", 0),
    DirichletBC("nodeset_3", 1),
    DirichletBC("nodeset_3", 2),
    DirichletBC("nodeset_5", 0),
    DirichletBC("nodeset_5", 1),
    DirichletBC("nodeset_5", 2)
]
neumann_bcs = []

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs)

##################
# ML setup
##################
def loss_function(params, problem, state_old, t, dt):
    field, physics, _ = params
    us = physics.vmap_field_values(field, problem.coords, t[0])
    pi, state_new = physics.potential_energy(
        physics, problem.domain, t, us, state_old, dt
    )
    return pi, dict(energy=pi, state_new=state_new)

    # (pi, state_new), R = physics.potential_energy_and_residual(
    #     params, problem.domain, t[0], us, state_old, dt[0]
    # )
    # return pi + 250.e9 * R, dict(
    #     energy=pi,
    #     residual=R,
    #     state_new=state_new
    # )
    # # return R, dict(
    # #     energy=pi,
    # #     residual=R,
    # #     state_new=state_new
    # # )

loss_function = UserDefinedLossFunction(loss_function)

# loss_function = PathDependentEnergyLoss()

params = Parameters(problem, key, seperate_networks=False)#, network_type=ResNet)
print(params)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, clip_gradients=False)
opt, opt_st = opt.init(params)

# for epoch in range(25000):
#     params, opt_st, loss = opt.step(params, opt_st, problem)

#     if epoch % 100 == 0:
#         print(f"Epoch = {epoch:7d} Energy = {loss[1]["energy"]:4.6f}")


# pp.init(
#     params,
#     problem,
#     f"output_{str(epoch).zfill(6)}.e",
#     node_variables=[
#         "field_values",
#         "internal_force"
#     ],
#     # element_variables=["deformation_gradients"],
#     element_variables=[
#         "deformation_gradient",
#         "state_variables"
#     ]
# )
# pp.write_outputs(params, problem)
# pp.close()


def _vmap_func(n):
    return problem.physics.constitutive_model.\
        initial_state()

ne = problem.domain.conns.shape[0]
nq = len(problem.domain.fspace.quadrature_rule)
state_old = vmap(vmap(_vmap_func))(jnp.zeros((ne, nq)))

pp.init(
    params, problem, output_file, 
    node_variables=[
        "field_values",
        "internal_force"
    ],
    element_variables=[
        "pk1_stress",
        "state_variables"
    ]
)

for n, t in enumerate(times[1:]):
    print(f"Load step {n + 1}")
    dt = t - times[n]
    print(f"Time step = {dt}")
    t = jnp.array([t])
    dt = jnp.array([dt])
    params = Parameters(problem, key, seperate_networks=False)#, network_type=ResNet)
    opt, opt_st = opt.init(params)


    for epoch in range(25000):
        params, opt_st, loss = opt.step(params, opt_st, problem, state_old, t, dt)

        if epoch % 100 == 0:
            print(f"Epoch = {epoch:7d} Energy = {loss[1]["energy"]:4.6f}")
            # print(f"Epoch = {epoch:7d} Energy = {loss[1]["energy"]:4.6f} Residual = {loss[1]["residual"]:4.6f}")

    # post process
    with nc.Dataset(output_file, "a") as dataset:
        pp.pp._write_step_outputs(dataset, n, params, problem, t[0], dt[0], state_old)

    state_old = loss[1]["state_new"]


