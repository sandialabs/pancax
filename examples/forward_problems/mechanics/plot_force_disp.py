from pancax.fem.read_exodus_mesh import _read_node_sets
import argparse
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

parser = argparse.ArgumentParser(description="A script to plot force displacement curves.")

parser.add_argument(
    "--exodus-file", 
    help="The name of the exodus file."
)
parser.add_argument(
    "--displacement-variable", 
    help="name of the displacment variable"
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

# exo_file = "../../../output_010000.e"
exo_file = args.exodus_file
# disp_var_name = "displ_y"
# force_var_name = "internal_force_y"
# nset_name = "nset_1"
disp_var_name = args.displacement_variable
force_var_name = args.force_variable
nset_name = args.nodeset

displacements = []
forces = []

with nc.Dataset(exo_file, "r") as data:
    nsets = _read_node_sets(data)
    node_var_names = data.variables["name_nod_var"]
    node_var_names.set_auto_mask(False)
    node_var_names = [b"".join(c).decode("UTF-8").rstrip() for c in node_var_names[:]]

    assert nset_name in nsets.keys()
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

plt.figure(1)
plt.plot(times, displacements)
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.savefig("time_disp.png")

plt.figure(2)
plt.plot(times, forces)
plt.xlabel("Time")
plt.ylabel("Force")
plt.savefig("time_force.png")

plt.figure(3)
plt.plot(displacements, forces)
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.savefig("disp_force.png")
