from pancax.fem.read_exodus_mesh import _read_node_sets
import netCDF4 as nc
import numpy as np
import pandas


with nc.Dataset("uniaxial_tension_hyperviscoelasticity.e", "r") as data:
    times = data.variables["time_whole"][:]
    nsets = _read_node_sets(data)

    node_var_names = data.variables["name_nod_var"]
    node_var_names.set_auto_mask(False)
    node_var_names = [b"".join(c).decode("UTF-8").rstrip() for c in node_var_names[:]]

    disp_x = data.variables[f"vals_nod_var{node_var_names.index("displ_x") + 1}"]
    disp_y = data.variables[f"vals_nod_var{node_var_names.index("displ_y") + 1}"]
    disp_z = data.variables[f"vals_nod_var{node_var_names.index("displ_z") + 1}"]

    
    # eventually set to node set nodes
    # nodes = np.arange(len(disp_x[0, :]))
    nodes = nsets["nodelist_1"]
    coords = np.array([
        data.variables["coordx"][:],
        data.variables["coordy"][:],
        data.variables["coordz"][:]
    ]).T
    coords = coords[nodes, :]

    data_all = []
    for n, time in enumerate(times):
        temp_disp = np.array([
            disp_x[n, nodes], disp_y[n, nodes], disp_z[n, nodes]
        ]).T
        temp_data = np.hstack([coords, time * np.ones((coords.shape[0], 1)), temp_disp])
        data_all.append(temp_data)
    
    data_all = np.vstack(data_all)
    column_names = ["x", "y", "z", "t", "u_x", "u_y", "u_z"]
    df = pandas.DataFrame(data_all, columns=column_names)
    print(df)
    df.to_csv("2holes_full_field_data.csv", index=False)
