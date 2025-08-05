def test_forward_domain():
    from pancax import VariationalDomain
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    times = jnp.linspace(0., 1.0, 2)
    VariationalDomain(mesh_file, times)


def test_forward_domain_tri3_p_order():
    from pancax import VariationalDomain
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent, 'mesh_10x.g')
    times = jnp.linspace(0., 1.0, 2)
    VariationalDomain(mesh_file, times, p_order=2)
