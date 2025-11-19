from pancax.fem.read_exodus_mesh import _read_node_sets
import argparse
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas

parser = argparse.ArgumentParser(
    description="A script to plot force displacement curves."
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
    "--pinn-exodus-file",
    help="The name of the PINN exodus file"
)
parser.add_argument(
    "--pinn-nodeset",
    help="name of the Sierra node set to read data from"
)
parser.add_argument(
    "--sierra-exodus-file",
    help="The name of the Sierra exodus file."
)
parser.add_argument(
    "--sierra-nodeset",
    help="name of the Sierra node set to read data from"
)

args = parser.parse_args()

disp_var_name = args.displacement_variable
force_var_name = args.force_variable
pinn_exo_file = args.pinn_exodus_file
pinn_nset_name = args.pinn_nodeset
sierra_exo_file = args.sierra_exodus_file
sierra_nset_name = args.sierra_nodeset


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


p_times, p_displacements, p_forces = \
    extract_data(pinn_exo_file, pinn_nset_name, disp_var_name, force_var_name)
s_times, s_displacements, s_forces = \
    extract_data(
        sierra_exo_file, sierra_nset_name, disp_var_name, force_var_name
    )

# save sierra exo file to csv file
df_data = np.array([p_times, p_displacements, p_forces]).T
df_columns = ["times", "disps", "forces"]
df = pandas.DataFrame(df_data, columns=df_columns)
df.to_csv("global_data.csv", index=False)


plt.figure(1)
plt.plot(p_times, p_displacements, label="PINN", linestyle="None", marker="o")
plt.plot(s_times, s_displacements, label="FEM")
plt.legend(loc="best")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.savefig("time_disp.png")

plt.figure(2)
plt.plot(p_times, p_forces, label="PINN", linestyle="None", marker="o")
plt.plot(s_times, s_forces, label="FEM")
plt.legend(loc="best")
plt.xlabel("Time")
plt.ylabel("Force")
plt.savefig("time_force.png")

plt.figure(3)
plt.plot(p_displacements, p_forces, label="PINN", linestyle="None", marker="o")
plt.plot(s_displacements, s_forces, label="FEM")
plt.legend(loc="best")
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.savefig("disp_force.png")
