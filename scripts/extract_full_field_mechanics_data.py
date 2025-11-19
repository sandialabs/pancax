from pancax.fem.read_exodus_mesh import _read_node_sets
import argparse
import netCDF4 as nc
import numpy as np
import pandas


parser = argparse.ArgumentParser(
    description="A script to full-field data "
    "responses from Sierra Solid Mechanics."
)

parser.add_argument(
    "--csv-file",
    help="name of a file to write data to in CSV format"
)
parser.add_argument(
    "--exodus-file",
    help="The name of the exodus file"
)
parser.add_argument(
    "--nodal-variables",
    help="name of the internal force variable"
)
parser.add_argument(
    "--nodeset",
    help="name of the node set to read data from"
)


args = parser.parse_args()

csv_file = args.csv_file
exo_file = args.exodus_file
nodal_var_names = args.nodal_variables.split(",")
nset_name = args.nodeset

print(f"Nodal var names = {nodal_var_names}")


def extract_nodal_data(exo_file, nset_name, var_names):
    vars = []
    with nc.Dataset(exo_file, "r") as data:
        nsets = _read_node_sets(data)
        node_var_names = data.variables["name_nod_var"]
        node_var_names.set_auto_mask(False)
        node_var_names = [
            b"".join(c).decode("UTF-8").rstrip() for c in node_var_names[:]
        ]

        assert nset_name in nsets.keys(), \
            f"Available nodesets are {nsets.keys()}"
        for var_name in var_names:
            assert var_name in node_var_names

        nset = nsets[nset_name]
        times = data.variables["time_whole"][:]
        coords = np.vstack((
            data.variables["coordx"][nset],
            data.variables["coordy"][nset],
            data.variables["coordz"][nset]
        ))

        for var_name in var_names:
            var_index = node_var_names.index(var_name)
            var_name = f"vals_nod_var{var_index + 1}"
            var_all_times = data.variables[var_name]
            temp_var = []
            for n, time in enumerate(times):
                var_vals = np.array(var_all_times[n, nset])
                temp_var.append(var_vals)
            temp_var = np.hstack(temp_var)
            vars.append(temp_var)

        n_time_steps = len(times)
        n_nodes = int(len(vars[0]) / n_time_steps)
        coord_vals = []
        time_vals = []
        for n, time in enumerate(times):
            coord_vals.append(coords)
            time_vals.append(time * np.ones((n_nodes,)))
        coord_vals = np.hstack(coord_vals)
        time_vals = np.hstack(time_vals)

    return np.vstack((time_vals, coord_vals, *vars)).T


df_data = extract_nodal_data(exo_file, nset_name, nodal_var_names)
df_columns = ["t", "x", "y", "z", *nodal_var_names]
df = pandas.DataFrame(df_data, columns=df_columns)
df.to_csv(csv_file, index=False)
