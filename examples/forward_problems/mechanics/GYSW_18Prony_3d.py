from pancax import *

##################
# for reproducibility
##################
key = random.PRNGKey(10)

##################
# file management
##################
mesh_file = find_mesh_file("2holes.g")
pp = PostProcessor(mesh_file, "exodus")

##################
# domain setup
##################
t1 = 100. # loading time
tend = t1 + 100. # relaxation time
times_1 = jnp.linspace(0., t1, 11)
times_2 = jnp.linspace(t1, tend, 11)
times = jnp.hstack((times_1, times_2[1:]))
domain = VariationalDomain(mesh_file, times, q_order=2)

##################
# physics setup
##################
# ramp followed by stress relaxation
def dirichlet_bc_func(xs, t, nn):
    length = 1.
    final_displacement = 1.
    # x, y, z = xs[0], xs[1], xs[2]
    y = xs[1]
    u_out = nn
    u_out = u_out.at[0].set(
        y * (y - length) * (t / tend) * nn[0] / length**2
    )

    u_out = jax.lax.cond(
        t > t1,
        lambda u: u.at[1].set(
            y * final_displacement / length
            + y * (y - length) * (t / tend) * nn[1] / length**2
        ),
        lambda u: u.at[1].set(
            y * (t / t1) * final_displacement / length
            + y * (y - length) * (t / tend) * nn[1] / length**2
        ),
        u_out
    )
    u_out = u_out.at[2].set(
        y * (y - length) * (t / tend) * nn[2] / length**2
    )
    return u_out

model = SimpleFeFv(
    Swanson(
      bulk_modulus=100.,
      A1=212.6772,
      P1=-0.13697328,
      B1=-28.9699,
      Q1=-0.02780753,
      C1=196.0172,
      R1=0.67121324,
      cutoff_strain=0.01
    ),
    PronySeries(
        moduli=[
            7.01E+00,
            7.54E+01,
            6.87E+01,
            1.15E+02,
            3.85E+02,
            2.21E+02,
            4.18E+01,
            1.82E+01,
            8.05E+00,
            3.91E+00,
            2.29E+00,
            1.45E+00,
            1.24E+00,
            8.78E-01,
            9.80E-01,
            4.75E-01,
            6.46E-01,
            8.77E-04
        ], 
        relaxation_times=[
            1.00E+10, #1.00E+50, # this number is outrageously large
            1.74E-14,
            1.97E-13,
            2.23E-12,
            2.52E-11,
            2.85E-10,
            3.22E-09,
            3.64E-08,
            4.12E-07,
            4.65E-06,
            5.26E-05,
            5.95E-04,
            6.73E-03,
            7.61E-02,
            8.60E-01,
            9.72E+00,
            1.10E+02,
            1.24E+03
        ]
    ),
    WLF(C1=17.44, C2=51.6, theta_ref=60.0),
)

model = SimpleFeFv(
    NeoHookean(bulk_modulus=100.0, shear_modulus=0.855),
    PronySeries(moduli=[1.0], relaxation_times=[10.0]),
    WLF(C1=17.44, C2=51.6, theta_ref=60.0),
)

# model = SimpleFeFv(
#     Swanson(
#         bulk_modulus=100.,
#         A1=2.08,
#         P1=-0.33,
#         B1=-0.0,
#         Q1=-0.5,
#         C1=0.0006,
#         R1=1.51,
#         cutoff_strain=0.00007
#     ),
#     PronySeries(moduli=[1.0], relaxation_times=[10.0]),
#     WLF(C1=17.44, C2=51.6, theta_ref=60.0),
# )		

# model = SimpleFeFv(
#     Swanson(
#         bulk_modulus=1500,
#         A1=212.6772,
#         P1=-0.13697328,
#         B1=-28.9699,
#         Q1=-0.02780753,
#         C1=196.0172,
#         R1=0.67121324,
#         cutoff_strain=0.01
#     ),
#     PronySeries(
#         moduli=[
#             #7.01E+00,
#             #7.54E+01,
#             #6.87E+01,
#             #1.15E+02,
#             #3.85E+02,
#             #2.21E+02,
#             #4.18E+01,
#             #1.82E+01,
#             #8.05E+00,
#             3.91E+00,
#             2.29E+00,
#             1.45E+00,
#             1.24E+00,
#             8.78E-01,
#             9.80E-01,
#             4.75E-01,
#             6.46E-01,
#             8.77E-04
#         ], 
#         relaxation_times=[
#             #1.00E+50,
#             #1.74E-14,
#             #1.97E-13,
#             #2.23E-12,
#             #2.52E-11,
#             #2.85E-10,
#             #3.22E-09,
#             #3.64E-08,
#             #4.12E-07,
#             4.65E-06,
#             5.26E-05,
#             5.95E-04,
#             6.73E-03,
#             7.61E-02,
#             8.60E-01,
#             9.72E+00,
#             1.10E+02,
#             1.24E+03                                             
#         ]
#     ),
#     WLF(C1=17.44, C2=51.6, theta_ref=23.0),
# )

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
loss_function = PathDependentEnergyLoss()
# loss_function = EnergyLoss()

params = Parameters(problem, key, seperate_networks=True)
print(params)

##################
# train network
##################
opt = Adam(loss_function, learning_rate=1.0e-3, has_aux=True, clip_gradients=False)
opt, opt_st = opt.init(params)
for epoch in range(500000):
    params, opt_st, loss = opt.step(params, opt_st, problem)

    if epoch % 10 == 0:
        print(epoch, flush=True)
        print(loss, flush=True)

    if epoch % 10000 == 0:
        print("writing exodus output")
        pp.init(
            params, problem,
            f"output_{str(epoch).zfill(6)}.e",
            node_variables=[
                "field_values",
                "internal_force"
            ],
            element_variables=[
                "state_variables"
            ]
        )
        pp.write_outputs(params, problem)
        pp.close()
