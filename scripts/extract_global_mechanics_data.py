from pancax.fem.read_exodus_mesh import _read_node_sets
import argparse
import netCDF4 as nc
import numpy as np
import pandas


parser = argparse.ArgumentParser(
    description="A script to extract global force displacement "
    "responses from Sierra Solid Mechanics."
)

parser.add_argument(
    "--csv-file",
    help="name of a file to write data to in CSV format"
)
parser.add_argument(
    "--displacement-variable",
    help="name of the displacment variable"
)
parser.add_argument(
    "--exodus-file",
    help="The name of the exodus file"
)
parser.add_argument(
    "--force-variable",
    help="name of the internal force variable"
)
parser.add_argument(
    "--nodeset",
    help="name of the node set to read data from"
)


args = parser.parse_args()

csv_file = args.csv_file
disp_var_name = args.displacement_variable
exo_file = args.exodus_file
force_var_name = args.force_variable
nset_name = args.nodeset


def extract_data(exo_file, nset_name, disp_var_name, force_var_name):
    displacements = []
    forces = []
    with nc.Dataset(exo_file, "r") as data:
        nsets = _read_node_sets(data)
        node_var_names = data.variables["name_nod_var"]
        node_var_names.set_auto_mask(False)
        node_var_names = [
            b"".join(c).decode("UTF-8").rstrip() for c in node_var_names[:]
        ]

        assert nset_name in nsets.keys(), \
            f"Available nodesets are {nsets.keys()}"
        assert disp_var_name in node_var_names
        assert force_var_name in node_var_names

        nset = nsets[nset_name]
        disp_var_index = node_var_names.index(disp_var_name)
        force_var_index = node_var_names.index(force_var_name)

        disp_var_name = f"vals_nod_var{disp_var_index + 1}"
        force_var_name = f"vals_nod_var{force_var_index + 1}"

        times = data.variables["time_whole"][:]

        disp_all_times = data.variables[disp_var_name]
        force_all_times = data.variables[force_var_name]

        for n, time in enumerate(times):
            disp = np.array(disp_all_times[n, nset])
            force = np.array(force_all_times[n, nset])

            displacements.append(np.mean(disp))
            forces.append(np.sum(force))

    times = np.array(times)
    displacements = np.array(displacements)
    forces = np.array(forces)

    return times, displacements, forces


times, displacements, forces = \
    extract_data(exo_file, nset_name, disp_var_name, force_var_name)

# save sierra exo file to csv file
df_data = np.array([times, displacements, forces]).T
df_columns = ["times", "disps", "forces"]
df = pandas.DataFrame(df_data, columns=df_columns)
df.to_csv(csv_file, index=False)
