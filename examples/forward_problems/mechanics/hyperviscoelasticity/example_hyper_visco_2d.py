from pancax import *

##################
# for reproducibility
##################
key = random.PRNGKey(10)

##################
# file management
##################
mesh_file = find_mesh_file("mesh_quad4.g")
pp = PostProcessor(mesh_file, "exodus")

##################
# domain setup
##################
# times = jnp.linspace(0.0, 2.0, 21)
times_1 = jnp.linspace(0., 1., 11)
times_2 = jnp.linspace(1., 11., 11)
times = jnp.hstack((times_1, times_2[1:]))
domain = VariationalDomain(mesh_file, times, q_order=2)

##################
# physics setup
##################
# dirichlet_bc_func = UniaxialTensionLinearRamp(
#     final_displacement=1.0, length=1.0, direction="y", n_dimensions=2
# )

# ramp followed by stress relaxation
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
model = SimpleFeFv(
    NeoHookean(bulk_modulus=10.0, shear_modulus=1.0),
    # PronySeries(moduli=[1.0, 2.0], relaxation_times=[10.0, 20.0]),
    PronySeries(moduli=[1.], relaxation_times=[10.]),
    WLF(C1=17.44, C2=51.6, theta_ref=60.0),
)
physics = SolidMechanics(model, PlaneStrain())
# physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)

ics = []
dirichlet_bcs = [
    DirichletBC("nset_1", 0),
    DirichletBC("nset_1", 1),
    # DirichletBC("nset_3", 2),
    DirichletBC("nset_3", 0),
    DirichletBC("nset_3", 1),
    # DirichletBC("nset_5", 2)
]
neumann_bcs = []

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs)

##################
# ML setup
##################
# loss_function = PathDependentEnergyLoss()
loss_function = EnergyLoss(is_path_dependent=True)

params = Parameters(
    problem, key,
    dirichlet_bc_func=dirichlet_bc_func,
    seperate_networks=False
)
print(params)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, clip_gradients=False)
opt, opt_st = opt.init(params)
for epoch in range(25000):
    params, opt_st, loss = opt.step(params, opt_st, problem)
    # logger.log_loss(loss, epoch)
    if epoch % 100 == 0:
        print(epoch, flush=True)
        print(loss, flush=True)
        # print(params.state[0, :, :, :])
        # print(params.physics.constitutive_model)

    if epoch % 10000 == 0:
        pp.init(
            params,
            problem,
            f"output_{str(epoch).zfill(6)}.e",
            node_variables=[
                "field_values",
                "internal_force"
            ],
            # element_variables=["deformation_gradients"],
            element_variables=[
                "deformation_gradient",
                "state_variables"
            ]
        )
        pp.write_outputs(params, problem)
        pp.close()
