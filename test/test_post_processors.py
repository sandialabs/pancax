from jax import random
from pancax import EssentialBC, VariationalDomain, NeoHookean, ThreeDimensional, SolidMechanics
from pancax import FieldPropertyPair, FixedProperties, MLP
from pancax import PostProcessor
from pathlib import Path
import jax
import jax.numpy as jnp
import os
import pytest


@pytest.fixture
def domain():
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
    return VariationalDomain(physics, essential_bcs, natural_bcs, mesh_file, times)
    

@pytest.fixture
def params():
    key = random.key(10)
    field_network = MLP(4, 3, 20, 3, jax.nn.tanh, key)
    props = FixedProperties([5., 0.3846])
    return FieldPropertyPair(field_network, props)


def test_post_processor(params, domain):
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    pp = PostProcessor(mesh_file)
    pp.init(domain, 'output.e',
        node_variables=[
            'displacement'
        ],
        element_variables=[
            'element_deformation_gradient'
        ]
    )
    pp.write_outputs(params, domain)
    pp.close()


def test_post_processor_bad_var_name(domain):
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    pp = PostProcessor(mesh_file)
    with pytest.raises(ValueError):
        pp.init(domain, 'output.e',
            node_variables=[
                'bad_var_name'
            ]
        )
