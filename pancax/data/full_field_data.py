from jaxtyping import Array
from typing import List
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
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

  # TODO still need to test this
  def plot_registration(self, domain):
    n_points_per_step = int(self.inputs.shape[0] / self.n_time_steps)
    first_step = self.inputs[:n_points_per_step, :]

    plt.figure(1)
    plt.scatter(first_step[:, 0], first_step[:, 1], color='blue', label='DIC')
    plt.scatter(domain.coords[:, 0], domain.coords[:, 1], color='red', label='Mesh')
    plt.xlabel('X coordinate (mm)')
    plt.ylabel('Y coordinate (mm)')
    plt.legend(loc='best')
    plt.savefig('dic_data_mesh_registration.png')
    print('here')
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
    if component == 'x':
      component = 0
    elif component == 'y':
      component = 1
    elif component == 'z':
      component = 2
    else:
      raise ValueError('Needs to be x, y, or z')

    set_inputs = self.inputs
    set_inputs = set_inputs.at[:, component].set(val)
    return eqx.tree_at(lambda x: x.inputs, self, set_inputs)
