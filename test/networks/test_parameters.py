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


def test_parameters(problem):
    from jax import random
    from pancax import Parameters, SolidMechanics
    key = random.PRNGKey(0)
    params = Parameters(problem, key=key)
    field, physics, state = params
    x = random.uniform(key=key, shape=(3,))
    t = random.uniform(key=key, shape=(1,))
    y = field(x, t)
    assert y.shape == (3,)
    assert type(physics) is SolidMechanics
    assert state is None


def test_parameters_freeze_fields_filter(problem):
    from jax import random
    from jax.tree_util import tree_leaves
    from pancax import Parameters
    key = random.PRNGKey(0)
    params = Parameters(problem, key=key)
    filter = params.freeze_fields_filter()

    assert all(x is False for x in tree_leaves(filter.fields))


def test_parameters_freeze_physics_filter(problem):
    from jax import random
    from jax.tree_util import tree_leaves
    from pancax import Parameters
    key = random.PRNGKey(0)
    params = Parameters(problem, key=key)
    filter = params.freeze_physics_filter()

    assert all(tree_leaves(filter.fields))
    assert all(x is False for x in tree_leaves(filter.physics))


def test_parameters_freeze_physics_normalization_filter(problem):
    from jax import random
    from pancax import Parameters
    key = random.PRNGKey(0)
    params = Parameters(problem, key=key)
    filter = params.freeze_physics_normalization_filter()

    # assert all(tree_leaves(filter.fields))

    assert filter.fields.x_mins is False
    assert filter.fields.x_maxs is False
    assert filter.fields.t_min is False
    assert filter.fields.t_max is False
