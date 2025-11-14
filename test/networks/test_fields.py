import pytest


@pytest.fixture
def problem():
    from pancax import (
        DirichletBC, ForwardProblem,
        NeoHookean, SolidMechanics, ThreeDimensional, VariationalDomain
    )
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent.parent, 'mesh.g')
    times = jnp.linspace(0., 1.0, 2)
    domain = VariationalDomain(mesh_file, times)

    physics = SolidMechanics(
        NeoHookean(bulk_modulus=0.833, shear_modulus=0.3846),
        ThreeDimensional()
    )
    ics = [
    ]
    # essential_bc_func = lambda x, t, z: z
    essential_bcs = [
        DirichletBC('nset_4', 0),
        DirichletBC('nset_4', 1),
        DirichletBC('nset_4', 2),
        DirichletBC('nset_6', 0),
        DirichletBC('nset_6', 1),
        DirichletBC('nset_6', 2)
    ]
    natural_bcs = [
    ]
    problem = ForwardProblem(domain, physics, ics, essential_bcs, natural_bcs)
    return problem


def test_fields(problem):
    from jax import random
    from pancax import Field
    key = random.key(0)
    field = Field(problem, key, seperate_networks=False)
    x = random.uniform(key=key, shape=(3,))
    t = random.uniform(key=key, shape=(1,))
    y = field(x, t)
    assert y.shape == (3,)


def test_fields_with_seperate_networks(problem):
    from jax import random
    from pancax import Field
    key = random.key(0)
    field = Field(problem, key, seperate_networks=True)
    x = random.uniform(key=key, shape=(3,))
    t = random.uniform(key=key, shape=(1,))
    y = field(x, t)
    assert y.shape == (3,)
