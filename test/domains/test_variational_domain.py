from pancax import EssentialBC, VariationalDomain
from pancax import NeoHookean, SolidMechanics, PlaneStrain, ThreeDimensional
from pathlib import Path
import jax.numpy as jnp
import os


def test_forward_domain():
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    essential_bc_func = lambda x, t, z: z
    essential_bcs = [
        EssentialBC('nset_4', 0),
        EssentialBC('nset_4', 1),
        EssentialBC('nset_4', 2),
        EssentialBC('nset_6', 0),
        EssentialBC('nset_6', 1),
        EssentialBC('nset_6', 2)
    ]
    natural_bcs = [
    ]
    physics = SolidMechanics(
        mesh_file, essential_bc_func, 
        NeoHookean(), ThreeDimensional()
    )
    times = jnp.linspace(0., 1.0, 2)
    domain = VariationalDomain(physics, essential_bcs, natural_bcs, mesh_file, times)


def test_forward_domain_tri3_p_order():
    mesh_file = os.path.join(Path(__file__).parent, 'mesh_10x.g')
    essential_bc_func = lambda x, t, z: z
    essential_bcs = [
        EssentialBC('nodeset_2', 0),
        EssentialBC('nodeset_2', 1),
        EssentialBC('nodeset_4', 0),
        EssentialBC('nodeset_4', 1)
    ]
    natural_bcs = [
    ]
    physics = SolidMechanics(
        mesh_file, essential_bc_func, 
        NeoHookean(), PlaneStrain()
    )
    times = jnp.linspace(0., 1.0, 2)
    domain = VariationalDomain(physics, essential_bcs, natural_bcs, mesh_file, times)

    domain = VariationalDomain(physics, essential_bcs, natural_bcs, mesh_file, times, p_order=2)

    # below is just to cover field_values
    field_values = domain.field_values
