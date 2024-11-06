from pancax.fem import read_exodus_mesh
import os


def test_read_exodus_mesh_hex8():
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mesh_hex8.g')
    mesh = read_exodus_mesh(f)


def test_read_exodus_mesh_quad4():
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mesh_quad4.g')
    mesh = read_exodus_mesh(f)


def test_read_exodus_mesh_quad9():
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mesh_quad9.g')
    mesh = read_exodus_mesh(f)


def test_read_exodus_mesh_tri():
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mesh_no_ssets.g')
    mesh = read_exodus_mesh(f)


def test_read_exodus_mesh_with_ssets_tri():
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mesh_1x.g')
    mesh = read_exodus_mesh(f)





