def test_simulation_times_unique_exception():
    from pancax.domains.base import BaseDomain
    from pancax.domains.base import \
        SimulationTimesNotUniqueException
    from pathlib import Path
    import jax.numpy as jnp
    import os
    import pytest
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    times = jnp.array([0., 0.])
    
    with pytest.raises(SimulationTimesNotUniqueException):
        BaseDomain(mesh_file, times)


def test_simulation_times_not_strictly_increasing():
    from pancax.domains.base import BaseDomain
    from pancax.domains.base import \
        SimulationTimesNotStrictlyIncreasingException
    from pathlib import Path
    import jax.numpy as jnp
    import os
    import pytest
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    times = jnp.array([0., -1.])

    with pytest.raises(SimulationTimesNotStrictlyIncreasingException):
        BaseDomain(mesh_file, times)
