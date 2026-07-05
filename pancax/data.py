from jaxtyping import Array
from typing import List, Optional, Union
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas


class FullFieldData(eqx.Module):
    """
    Data structure to store full field data used as ground truth
    for output fields of a PINN when solving inverse problems.

    :param inputs: Data that serves as inputs to the PINN
    :param outputs: Data that serves as outputs of the PINN
    :param n_time_steps: Variable used for book keeping
    """

    inputs: Array
    outputs: Array
    n_time_steps: int

    def __init__(
        self,
        data_file: str,
        input_keys: List[str],
        output_keys: List[str]
    ):
        df = pandas.read_csv(data_file)
        df.columns = df.columns.str.strip()
        self.inputs = jnp.array(df[input_keys].values)
        self.outputs = jnp.array(df[output_keys].values)
        self.n_time_steps = len(jnp.unique(self.inputs[:, -1]))

    def __len__(self):
        return self.inputs.shape[0]

    # TODO still need to test this
    def plot_registration(self, domain):
        n_points_per_step = int(self.inputs.shape[0] / self.n_time_steps)
        first_step = self.inputs[:n_points_per_step, :]

        plt.figure(1)
        plt.scatter(
            first_step[:, 0], first_step[:, 1],
            color="blue", label="DIC"
        )
        plt.scatter(
            domain.coords[:, 0], domain.coords[:, 1],
            color="red", label="Mesh"
        )
        plt.xlabel("X coordinate (mm)")
        plt.ylabel("Y coordinate (mm)")
        plt.legend(loc="best")
        plt.savefig("dic_data_mesh_registration.png")
        print("here")
        # plt.clf()

    def plot_data(self, domain, time_step):
        pass

    def shift_inputs(self, x, y, z):
        shifted_inputs = self.inputs
        shifted_inputs = shifted_inputs.at[:, 0].set(shifted_inputs[:, 0] + x)
        shifted_inputs = shifted_inputs.at[:, 1].set(shifted_inputs[:, 1] + y)
        shifted_inputs = shifted_inputs.at[:, 2].set(shifted_inputs[:, 2] + z)
        return eqx.tree_at(lambda x: x.inputs, self, shifted_inputs)

    def set_input_component_values(self, component, val):
        if component == "x":
            component = 0
        elif component == "y":
            component = 1
        elif component == "z":
            component = 2
        else:
            raise ValueError("Needs to be x, y, or z")

        set_inputs = self.inputs
        set_inputs = set_inputs.at[:, component].set(val)
        return eqx.tree_at(lambda x: x.inputs, self, set_inputs)


class FullFieldDataLoader(eqx.Module):
    data: FullFieldData
    indices: np.ndarray

    def __init__(self, data: FullFieldData) -> None:
        self.data = data
        self.indices = np.arange(len(self.data))

    def __len__(self):
        return len(self.data)

    def dataloader(self, batch_size: int):
        perm = np.random.permutation(self.indices)
        start = 0
        end = batch_size
        while end <= len(self):
            batch_perm = perm[start:end]
            yield self.data.inputs[batch_perm], self.data.outputs[batch_perm]
            start = end
            end = start + batch_size


class GlobalDataTimesNotUniqueException(Exception):
    pass


class GlobalDataTimesNotStrictlyIncreasingException(Exception):
    pass


# TODO currently hardcoded to force which may be limiting
# for others interested in doing other physics
class GlobalData(eqx.Module):
    """
    Data structure that holds global data to be used as
    ground truth for some global field calculated from
    PINN outputs used in inverse modeling training

    :param times: A set of times used to compare to physics calculations
    :param displacements: Currently hardcoded to \
        use a displacement-force curve TODO
    :param outputs: Field used as ground truth, hardcoded \
        essentially to a reaction force now
    :param n_nodes: Book-keeping variable for number of nodes \
        on nodeset to measure global response from
    :param n_time_steps: Book-keeping variable
    :param reaction_nodes: Node set nodes for where to measure reaction forces
    :param reaction_dof: Degree of freedom to use \
        for reaction force calculation
    """

    times: Array  # change to inputs?
    displacements: Array
    outputs: Array
    n_nodes: int
    n_time_steps: int
    reaction_nodes: Array
    reaction_dof: int

    def __init__(
        self,
        data_file: str,
        times_key: str,
        disp_key: str,
        force_key: str,
        mesh_file: str,
        nset_id: int,
        reaction_dof: Union[int, str],
        n_time_steps: int,
        plotting: Optional[bool] = False,
        interpolate: Optional[bool] = False
    ):
        # read in data
        df = pandas.read_csv(data_file)
        df.columns = df.columns.str.strip()
        times_in = df[times_key].values
        disps_in = df[disp_key].values
        forces_in = df[force_key].values
        # interpolate data onto times
        if interpolate:
            times = \
                np.linspace(np.min(times_in), np.max(times_in), n_time_steps)
        else:
            times = times_in

        # checking to make sure time values are unique
        if len(times) != len(set(times)):
            raise GlobalDataTimesNotUniqueException()

        # checking provided times are strictly increasing
        for i in range(len(times) - 1):
            if times[i] >= times[i + 1]:
                raise GlobalDataTimesNotStrictlyIncreasingException()

        # interpolating disp/force
        # TODO make this not so mechanics centric
        disp_interp = np.interp(times, times_in, disps_in)
        force_interp = np.interp(times, times_in, forces_in)

        if plotting:
            plt.figure(1)
            plt.plot(times_in, disps_in, label="Raw Data")
            plt.plot(
                times, disp_interp,
                label="Interpolated", linestyle="None", marker="o"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Displacement (mm)")
            plt.savefig("mts_time_displacement.png")
            # plt.clf()

            plt.figure(2)
            plt.plot(times_in, forces_in, label="Raw Data")
            plt.plot(
                times, force_interp,
                label="Interpolated", linestyle="None", marker="o"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.savefig("mts_time_force.png")
            # plt.clf()

            plt.figure(3)
            plt.plot(disps_in, forces_in, label="Raw Data")
            plt.plot(
                disp_interp,
                force_interp,
                label="Interpolated",
                linestyle="None",
                marker="o",
            )
            plt.xlabel("Displacement (mm)")
            plt.ylabel("Force (N)")
            plt.savefig("mts_displacement_force.png")
            # plt.clf()

        with nc.Dataset(mesh_file, "r") as dataset:
            nodes = dataset.variables[f"node_ns{nset_id}"][:] - 1
            reaction_nodes = jnp.array(nodes)
            n_nodes = len(reaction_nodes)

        if reaction_dof == "x":
            reaction_dof = 0
        elif reaction_dof == "y":
            reaction_dof = 1
        elif reaction_dof == "z":
            reaction_dof = 2
        else:
            raise ValueError("reaction_dof needs to be either x or y.")

        # set things
        self.times = jnp.array(times)
        self.displacements = jnp.array(disp_interp)
        self.outputs = jnp.array(force_interp)
        self.n_nodes = n_nodes
        self.n_time_steps = len(times)
        self.reaction_nodes = reaction_nodes
        self.reaction_dof = reaction_dof
