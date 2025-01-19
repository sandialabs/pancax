from jaxtyping import Array
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas


# TODO currently hardcoded to force which may be limiting
# for others interested in doing other physics
class GlobalData(eqx.Module):
  """
  Data structure that holds global data to be used as
  ground truth for some global field calculated from
  PINN outputs used in inverse modeling training

  :param times: A set of times used to compare to physics calculations
  :param displacements: Currently hardcoded to use a displacement-force curve TODO
  :param outputs: Field used as ground truth, hardcoded essentially to a reaction force now
  :param n_nodes: Book-keeping variable for number of nodes on nodeset to measure global response from
  :param n_time_steps: Book-keeping variable
  :param reaction_nodes: Node set nodes for where to measure reaction forces
  :param reaction_dof: Degree of freedom to use for reaction force calculation
  """
  times: Array # change to inputs?
  displacements: Array
  outputs: Array
  n_nodes: int
  n_time_steps: int
  reaction_nodes: Array
  reaction_dof: int
  
  def __init__(
    self,
    data_file: str, times_key: str, disp_key: str, force_key: str,
    mesh_file: str, nset_id: int, reaction_dof: Union[int, str], 
    n_time_steps: int,
    plotting: Optional[bool] = False
  ):
    # read in data
    df = pandas.read_csv(data_file)  
    df.columns = df.columns.str.strip()
    times_in = df[times_key].values
    disps_in = df[disp_key].values
    forces_in = df[force_key].values
    # interpolate data onto times
    times = np.linspace(np.min(times_in), np.max(times_in), n_time_steps)
    disp_interp = np.interp(times, times_in, disps_in)
    force_interp = np.interp(times, times_in, forces_in)

    if plotting:
      plt.figure(1)
      plt.plot(times_in, disps_in, label='Raw Data')
      plt.plot(times, disp_interp, label='Interpolated', linestyle='None', marker='o')
      plt.xlabel('Time (s)')
      plt.ylabel('Displacement (mm)')
      plt.savefig('mts_time_displacement.png')
      # plt.clf()

      plt.figure(2)
      plt.plot(times_in, forces_in, label='Raw Data')
      plt.plot(times, force_interp, label='Interpolated', linestyle='None', marker='o')
      plt.xlabel('Time (s)')
      plt.ylabel('Force (N)')
      plt.savefig('mts_time_force.png')
      # plt.clf()

      plt.figure(3)
      plt.plot(disps_in, forces_in, label='Raw Data')
      plt.plot(disp_interp, force_interp, label='Interpolated', linestyle='None', marker='o')
      plt.xlabel('Displacement (mm)')
      plt.ylabel('Force (N)')
      plt.savefig('mts_displacement_force.png')
      # plt.clf()

    with nc.Dataset(mesh_file, 'r') as dataset:
      nodes = dataset.variables[f'node_ns{nset_id}'][:] - 1
      reaction_nodes = jnp.array(nodes)
      n_nodes = len(reaction_nodes)


    if reaction_dof == 'x':
      reaction_dof = 0
    elif reaction_dof == 'y':
      reaction_dof = 1
    elif reaction_dof == 'z':
      reaction_dof = 2
    else:
      raise ValueError('reaction_dof needs to be either x or y.')

    # set things
    self.times = jnp.array(times)
    self.displacements = jnp.array(disp_interp)
    self.outputs = jnp.array(force_interp)
    self.n_nodes = n_nodes
    self.n_time_steps = len(times)
    self.reaction_nodes = reaction_nodes
    self.reaction_dof = reaction_dof
