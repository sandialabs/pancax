def test_find_data_file():
    from pancax import find_data_file
    find_data_file("data_global.csv")


def test_find_data_file_not_found():
    from pancax import find_data_file
    from pancax.utils import DataFileNotFoundException
    import pytest

    with pytest.raises(DataFileNotFoundException):
        find_data_file("bad_file_name.csv")


def test_find_mesh_file():
    from pancax import find_mesh_file
    find_mesh_file("mesh.g")


def test_find_mesh_file_not_found():
    from pancax import find_mesh_file
    from pancax.utils import MeshFileNotFoundException
    import pytest

    with pytest.raises(MeshFileNotFoundException):
        find_mesh_file("bad_mesh_file.g")
