from pancax import CollocationDomain, EssentialBC, NaturalBC
from pancax import NeoHookean, SolidMechanics, PlaneStrain, ThreeDimensional
from pathlib import Path
import jax.numpy as jnp
import os
import pytest


def test_collocation_domain_3D():
    with pytest.raises(ValueError):
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
        domain = CollocationDomain(physics, essential_bcs, natural_bcs, mesh_file, times)


def test_collocation_domain_tri3_p_order():
    mesh_file = os.path.join(Path(__file__).parent, 'mesh_10x.g')
    essential_bc_func = lambda x, t, z: z
    essential_bcs = [
        EssentialBC('nodeset_2', 0),
        EssentialBC('nodeset_2', 1),
        EssentialBC('nodeset_4', 0),
        EssentialBC('nodeset_4', 1)
    ]
    natural_bcs = [
        NaturalBC('sideset_3', lambda x, t: 0.),
        NaturalBC('sideset_4', lambda x, t: 0.),
    ]
    physics = SolidMechanics(
        mesh_file, essential_bc_func, 
        NeoHookean(), PlaneStrain()
    )
    times = jnp.linspace(0., 1.0, 2)

    domain = CollocationDomain(physics, essential_bcs, natural_bcs, mesh_file, times, p_order=2)

    # below is just to cover field_values
    field_values = domain.field_values
