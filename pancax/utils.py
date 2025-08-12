from pathlib import Path
import inspect
import os


class DataFileNotFoundException(Exception):
    pass


class MeshFileNotFoundException(Exception):
    pass


def find_data_file(data_file_in: str):
    call_file = Path(inspect.stack()[1].filename)
    call_file_dir = call_file.parent

    data_file = Path(os.path.join(call_file_dir, data_file_in))

    if data_file.is_file():
        print(f"Found {data_file_in} in {data_file.parent}")
        return data_file

    data_file = Path(os.path.join(call_file_dir, "data", data_file_in))

    if data_file.is_file():
        print(f"Found {data_file_in} in {data_file.parent}")
        return data_file

    raise DataFileNotFoundException(
        f"Could not find data file {data_file_in} in either "
        f"{call_file_dir} or {call_file_dir}/data"
    )


def find_mesh_file(mesh_file_in: str):
    call_file = Path(inspect.stack()[1].filename)
    call_file_dir = call_file.parent

    mesh_file = Path(os.path.join(call_file_dir, mesh_file_in))

    if mesh_file.is_file():
        print(f"Found {mesh_file_in} in {mesh_file.parent}")
        return mesh_file

    mesh_file = Path(os.path.join(call_file, "mesh", mesh_file_in))

    if mesh_file.is_file():
        print(f"Found {mesh_file_in} in {mesh_file.parent}")
        return mesh_file

    raise MeshFileNotFoundException(
        f"Could not find data file {mesh_file_in} in either "
        f"{call_file_dir} or {call_file_dir}/mesh"
    )
