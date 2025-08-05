import pytest


@pytest.fixture
def mlp():
    from pancax import MLP
    import jax
    return MLP(2, 3, 2, 2, jax.nn.tanh, key=jax.random.PRNGKey(0))


@pytest.fixture
def trunc_init():
    import pancax
    return pancax.networks.base.trunc_init


@pytest.fixture
def uniform_init():
    import pancax
    return pancax.networks.base.uniform_init


@pytest.fixture
def zero_init():
    import pancax
    return pancax.networks.base.zero_init


@pytest.fixture
def models(mlp):
    return [mlp]


@pytest.fixture
def init_funcs(
    trunc_init,
    uniform_init,
    zero_init
):
    return [
        trunc_init,
        uniform_init,
        zero_init
    ]


def test_serde(models):
    import os
    eqx_file_base_name = "temp"
    for model in models:
        model.serialise(eqx_file_base_name, 1)
        new_model = model.deserialise(f"{eqx_file_base_name}_0000001.eqx")
        assert model == new_model
    os.system(f"rm -f {eqx_file_base_name}_0000001.eqx")


def test_uniform_init(models, init_funcs):
    import jax
    import numpy as np

    key = jax.random.PRNGKey(np.random.randint(0, 1000))

    for model in models:
        for init_func in init_funcs:
            new_model = model.init(init_func, key=key)
