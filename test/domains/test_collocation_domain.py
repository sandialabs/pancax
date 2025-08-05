def test_collocation_domain_3D():
    from pancax import CollocationDomain
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    times = jnp.linspace(0., 1.0, 2)
    CollocationDomain(mesh_file, times)


def test_collocation_domain_tri3_p_order():
    from pancax import CollocationDomain
    from pathlib import Path
    import jax.numpy as jnp
    import os
    mesh_file = os.path.join(Path(__file__).parent, 'mesh_10x.g')
    times = jnp.linspace(0., 1.0, 2)
    CollocationDomain(mesh_file, times, p_order=2)
