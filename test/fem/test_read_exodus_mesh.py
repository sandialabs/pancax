def test_read_exodus_mesh_hex8():
    from pancax.fem import read_exodus_mesh
    import os
    f = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mesh_hex8.g"
    )
    read_exodus_mesh(f)


def test_read_exodus_mesh_quad4():
    from pancax.fem import read_exodus_mesh
    import os
    f = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mesh_quad4.g"
    )
    read_exodus_mesh(f)


def test_read_exodus_mesh_quad9():
    from pancax.fem import read_exodus_mesh
    import os
    f = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mesh_quad9.g"
    )
    read_exodus_mesh(f)


def test_read_exodus_mesh_tri():
    from pancax.fem import read_exodus_mesh
    import os
    f = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mesh_no_ssets.g"
    )
    read_exodus_mesh(f)


def test_read_exodus_mesh_with_ssets_tri():
    from pancax.fem import read_exodus_mesh
    import os
    f = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mesh_1x.g"
    )
    read_exodus_mesh(f)
