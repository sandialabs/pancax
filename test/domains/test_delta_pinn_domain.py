def test_forward_domain():
    from pancax import DeltaPINNDomain
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    # essential_bc_func = lambda x, t, z: z
    # essential_bcs = [
    #     EssentialBC('nset_4', 0),
    #     EssentialBC('nset_4', 1),
    #     EssentialBC('nset_4', 2),
    #     EssentialBC('nset_6', 0),
    #     EssentialBC('nset_6', 1),
    #     EssentialBC('nset_6', 2)
    # ]
    # natural_bcs = [
    # ]
    # physics = SolidMechanics(
    #     mesh_file, essential_bc_func,
    #     NeoHookean(), ThreeDimensional(),
    #     use_delta_pinn=True
    # )
    times = jnp.linspace(0., 1.0, 2)
    # domain = DeltaPINNDomain(
    # physics, essential_bcs, natural_bcs, mesh_file, times, 20)
    DeltaPINNDomain(mesh_file, times, n_eigen_values=100)
