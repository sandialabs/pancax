from pancax import *

##################
# for reproducibility
##################
key = random.key(10)

##################
# file management
##################
mesh_file = find_mesh_file('mesh_quad4_coarse.g')
logger = Logger('pinn.log', log_every=250)
pp = PostProcessor(mesh_file, 'exodus')

##################
# domain setup
##################
times = jnp.linspace(0.0, 1.0, 11)
domain = VariationalDomain(mesh_file, times, q_order=2)

##################
# physics setup
##################
dirichlet_bc_func = UniaxialTensionLinearRamp(
  final_displacement=1.0, length=1.0, direction='y', n_dimensions=2
)
model = SimpleFeFv(
  NeoHookean(
    bulk_modulus=1000., 
    shear_modulus=0.855
  ),
  PronySeries(
    moduli=[1.], 
    relaxation_times=[10.]
  ),
  WLF(
    C1=17.44, 
    C2=51.6, 
    theta_ref=60.
  ),
)
physics = SolidMechanics(model, PlaneStrain())
physics = physics.update_dirichlet_bc_func(dirichlet_bc_func)

ics = [
]
dirichlet_bcs = [
  DirichletBC('nset_1', 0),
  DirichletBC('nset_1', 1),
  DirichletBC('nset_3', 0),
  DirichletBC('nset_3', 1),
]
neumann_bcs = [
]

##################
# problem setup
##################
problem = ForwardProblem(domain, physics, ics, dirichlet_bcs, neumann_bcs)

##################
# ML setup
##################
loss_function = EnergyLoss()
params = Parameters(problem, key, seperate_networks=True)
print(params)

##################
# train network
##################
