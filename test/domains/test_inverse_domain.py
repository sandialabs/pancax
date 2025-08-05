# from pancax import EssentialBC, InverseDomain
# from pancax import NeoHookean, SolidMechanics, PlaneStrain, ThreeDimensional
# from pancax import FullFieldData, GlobalData
# from pathlib import Path
# import jax.numpy as jnp
# import os


# def test_inverse_domain():
#     field_data_file = os.path.join(
# Path(__file__).parent, 'data_full_field.csv')
#     global_data_file = os.path.join(Path(__file__).parent, 'data_global.csv')
#     mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
#     essential_bc_func = lambda x, t, z: z
#     essential_bcs = [
#         EssentialBC('nset_4', 0),
#         EssentialBC('nset_4', 1),
#         EssentialBC('nset_4', 2),
#         EssentialBC('nset_6', 0),
#         EssentialBC('nset_6', 1),
#         EssentialBC('nset_6', 2)
#     ]
#     natural_bcs = [
#     ]
#     physics = SolidMechanics(
#         mesh_file, essential_bc_func,
#         NeoHookean(), ThreeDimensional()
#     )
#     full_field_data = FullFieldData(
#         field_data_file,
#         input_keys=['x', 'y', 'z', 't'],
#         output_keys=['u_x', 'u_y', 'u_z']
#     )
#     global_data = GlobalData(
#         global_data_file,
#         times_key='t',
#         disp_key='u_x',
#         force_key='f_x',
#         mesh_file=mesh_file,
#         nset_id=6,
#         reaction_dof='x',
#         n_time_steps=11
#     )
#     domain = InverseDomain(
#         physics, essential_bcs, natural_bcs, mesh_file, global_data.times,
#         full_field_data, global_data
#     )
