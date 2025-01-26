from jax import random
from pancax import DirichletBC, VariationalDomain, NeoHookean, ThreeDimensional, SolidMechanics
from pancax import FieldPhysicsPair, MLP
from pancax import PostProcessor, ForwardProblem
from pathlib import Path
import jax
import jax.numpy as jnp
import os
import pytest


@pytest.fixture
def problem():
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    times = jnp.linspace(0., 1.0, 2)
    domain = VariationalDomain(mesh_file, times)

    physics = SolidMechanics(NeoHookean(bulk_modulus=0.833, shear_modulus=0.3846), ThreeDimensional())
    ics = [
    ]
    essential_bc_func = lambda x, t, z: z
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


@pytest.fixture
def params(problem):
    key = random.key(10)
    field_network = MLP(4, 3, 20, 3, jax.nn.tanh, key)
    return FieldPhysicsPair(field_network, problem.physics)


def test_post_processor(params, problem):
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    pp = PostProcessor(mesh_file)
    pp.init(problem, 'output.e',
        node_variables=[
            'field_values'
        ],
        # element_variables=[
        #     'element_deformation_gradient'
        # ]
    )
    pp.write_outputs(params, problem)
    pp.close()


def test_post_processor_bad_var_name(problem):
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    pp = PostProcessor(mesh_file)
    with pytest.raises(ValueError):
        pp.init(problem, 'output.e',
            node_variables=[
                'bad_var_name'
            ]
        )
