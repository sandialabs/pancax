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
    plt.clf()

  def plot_data(self, domain, time_step):
    pass
