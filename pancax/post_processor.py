from pancax.physics import incompressible_internal_force, internal_force
from typing import List, Optional
import exodus3 as exodus
import jax
import os
import numpy as onp


class PostProcessor:
  def __init__(self, mesh_file: str) -> None:
    self.mesh_file = mesh_file
    self.exo = None
    self.node_variables = None
    self.element_variables = None

  def check_variable_names(self, domain, variables) -> None:
    for var in variables:
      if var == 'internal_force' or var == 'incompressible_internal_force':
        continue

      if var not in domain.physics.var_name_to_method.keys():
        str = f'Unsupported variable requested for output {var}.\n'
        str += f'Supported variables include:\n'
        for v in domain.physics.var_name_to_method.keys():
          str += f'  {v}\n'
        raise ValueError(str)

  def close(self) -> None:
    self.exo.close()

  def copy_mesh(self, output_file: str) -> None:
    if os.path.isfile(output_file):
      os.remove(output_file)

    exo_temp = exodus.copy_mesh(self.mesh_file, output_file)
    exo_temp.close()
    self.exo = exodus.exodus(output_file, mode='a', array_type='numpy')

  def get_node_variable_number(self, domain, variables) -> int:
    n = 0
    for var in variables:
      if var == 'internal_force' or var == 'incompressible_internal_force':
        n = n + len(domain.physics.field_value_names)
      else:
        n = n + len(domain.physics.var_name_to_method[var]['names'])
    
    return n

  def get_element_variable_number(self, domain, variables) -> int:
    n = 0
    for var in variables:
      n = n + len(domain.physics.var_name_to_method[var]['names'])
    if n > 0:
      q_points = len(domain.fspace.quadrature_rule)
      return n * q_points
    else:
      return 0

  def init(
    self, 
    domain, output_file: str, 
    node_variables: List[str],
    element_variables: Optional[List[str]] = []
  ) -> None:
    self.copy_mesh(output_file)
    self.check_variable_names(domain, node_variables)
    self.check_variable_names(domain, element_variables)
    self.node_variables = node_variables
    self.element_variables = element_variables
    self.exo.set_node_variable_number(
      self.get_node_variable_number(domain, node_variables)
    )
    self.exo.set_element_variable_number(
      self.get_element_variable_number(domain, element_variables)
    )
    n = 1
    for var in self.node_variables:
      if var == 'internal_force' or var == 'incompressible_internal_force':
        self.exo.put_node_variable_name('internal_force_x', n)
        self.exo.put_node_variable_name('internal_force_y', n + 1)
        self.exo.put_node_variable_name('internal_force_z', n + 2)
        n = n + 3
      else:
        for v in domain.physics.var_name_to_method[var]['names']:
          self.exo.put_node_variable_name(v, n)
          n = n + 1

    if len(element_variables) > 0:
      q_points = len(domain.fspace.quadrature_rule)
      n = 1
      for var in self.element_variables:
        for v in domain.physics.var_name_to_method[var]['names']:
          for q in range(q_points):
            name = f'{v}_{q + 1}'
            self.exo.put_element_variable_name(name, n)
            n = n + 1

  def index_to_component(self, index):
    if index == 0:
      string = 'x'
    elif index == 1:
      string = 'y'
    elif index == 2:
      string = 'z'
    else:
      raise ValueError('Should be 0, 1, or 2')
    return string

  def write_outputs(self, params, domain) -> None:
    physics = domain.physics
    times = domain.times
    for n, time in enumerate(times):
      self.exo.put_time(n + 1, time)

      for var in self.node_variables:
        if var == 'internal_force' or var == 'incompressible_internal_force':
          us = jax.vmap(physics.field_values, in_axes=(None, 0, None))(params.fields, domain.coords, time)
          fs = onp.array(internal_force(domain, us, params.properties()))
          for i in range(fs.shape[1]):
            self.exo.put_node_variable_values(f'internal_force_{self.index_to_component(i)}', n + 1, fs[:, i])
        else:
          output = physics.var_name_to_method[var]
          pred = onp.array(output['method'](params, domain, time))
          if len(pred.shape) > 2:
            for i in range(pred.shape[1]):
              for j in range(pred.shape[2]):
                k = pred.shape[1] * i + j
                self.exo.put_node_variable_values(output['names'][k], n + 1, pred[:, i, j])
          else:
            for i in range(pred.shape[1]):
              self.exo.put_node_variable_values(output['names'][i], n + 1, pred[:, i])

      if len(self.element_variables) > 0:
        n_q_points = len(domain.fspace.quadrature_rule)
        for var in self.element_variables:
          output = physics.var_name_to_method[var]
          pred = onp.array(output['method'](params, domain, time))
          for q in range(n_q_points):
            for i in range(pred.shape[2]):
              name = f'{output["names"][i]}_{q + 1}'
              # NOTE this will only work on a single block 
              self.exo.put_element_variable_values(1, name, n + 1, pred[:, q, i])
