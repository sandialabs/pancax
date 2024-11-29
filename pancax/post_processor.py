from pancax.physics import incompressible_internal_force, internal_force
from typing import List, Optional

try:
  import exodus3 as exodus
except:
  print('WARNING: Could not find exodus3 module. You\'ll need to use vtk output')

try:
  import vtk
except:
  print('WARNING: Could not find vtk module. You\'ll need to use another form of output')

import jax
import os
import numpy as onp
# import vtk


class ExodusPostProcessor:
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
    element_variables: List[str]
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


class VtkPostProcessor:
  def __init__(self, mesh_file: str) -> None:
    self.mesh_file = mesh_file
    self.node_variables = None
    self.element_variables = None
    self.vtk_points = None
    self.poly_data = None
    self.writer = vtk.vtkXMLMultiBlockDataWriter()

  def close(self):
    pass 

  def init(
    self,
    domain, output_file: str,
    node_variables,
    element_variables
  ) -> None:
    
    self.node_variables = node_variables
    self.element_variables = element_variables

    # self.convert_exodus_to_xml(domain.mesh_file, output_file)
    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(domain.mesh_file)
    reader.Update()

    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(reader.GetOutput())
    writer.Write()
    assert False

    # exodus_reader = vtk.vtkExodusIIReader()
    # exodus_reader.SetFileName(domain.mesh_file)
    # exodus_reader.Update()
    # # output = exodus_reader.GetOutput()
    # # print(exodus_reader.GetOutput())
    # self.poly_data = exodus_reader.GetOutput()
    # # writer = vtk.vtkXMLMultiBlockDataWriter()
    # self.writer.SetFileName(output_file)
    # writer.SetInputData(exodus_reader.GetOutput())
    # writer.Write()

    # delete this
    # writer.SetInputData(poly_data)
    # writer.Write()

  def write_outputs(self, params, domain) -> None:
    physics = domain.physics
    times = domain.times

    multi_block = vtk.vtkMultiBlockDataSet()
    for n, time in enumerate(times):
      # self.exo.put_time(n + 1, time)

      # poly_data = vtk.vtkPolyData()
      # poly_data.SetPoints(self.vtk_points)

      for var in self.node_variables:
        vector_array = vtk.vtkDoubleArray()
        vector_array.SetName(var)

        if var == 'internal_force' or var == 'incompressible_internal_force':
          assert False
          us = jax.vmap(physics.field_values, in_axes=(None, 0, None))(params.fields, domain.coords, time)
          fs = onp.array(internal_force(domain, us, params.properties()))
          # for i in range(fs.shape[1]):
          #   self.exo.put_node_variable_values(f'internal_force_{self.index_to_component(i)}', n + 1, fs[:, i])
        else:
          output = physics.var_name_to_method[var]
          pred = onp.array(output['method'](params, domain, time))
          if len(pred.shape) > 2:
            assert False, 'still need to hook this up'
          #   for i in range(pred.shape[1]):
          #     for j in range(pred.shape[2]):
          #       k = pred.shape[1] * i + j
          #       self.exo.put_node_variable_values(output['names'][k], n + 1, pred[:, i, j])
          else:
            # vector_array.SetNumberOfComponents(pred.shape[1])
            vector_array.SetNumberOfComponents(3)

            for i in range(pred.shape[0]):
              if pred.shape[1] == 2:
                vector_array.InsertNextTuple(tuple(pred[i, :]) + (0.,))
              elif pred.shape[2] == 3:
                vector_array.InsertNextTuple(tuple(pred[i, :]))
              else:
                assert False

            # self.poly_data.GetBlock(n).SetVectors(vector_array)
            self.poly_data.GetFieldData().AddArray(vector_array)

            # time_array = vtk.vtkDoubleArray()
            # time_array.SetName('Times')
            # time_array.InsertNextValue(n)

            multi_block.SetBlock(n, self.poly_data)

            # time_array = vtk.vtkDoubleArray()
            # time_array.SetName('TimeValues')
            # time_array.InsertNextValue(n)  # Store the time step index or actual time value
            multi_block.GetMetaData(n).Set('Time', time)

            # multi_block.GetMetaData(n).Set(time_array)
            # multi_block.GetMetaData(n).Set(vtk.vtkCompositeDataSet.NAME(), f'Time Step {n}')
            # multi_block.GetMetaData(n).Set(vtk.vtkCompositeDataSet.TIME(), n)

            #   self.exo.put_node_variable_values(output['names'][i], n + 1, pred[:, i])

      # if len(self.element_variables) > 0:
      #   n_q_points = len(domain.fspace.quadrature_rule)
      #   for var in self.element_variables:
      #     output = physics.var_name_to_method[var]
      #     pred = onp.array(output['method'](params, domain, time))
      #     for q in range(n_q_points):
      #       for i in range(pred.shape[2]):
      #         name = f'{output["names"][i]}_{q + 1}'
      #         # NOTE this will only work on a single block 
      #         self.exo.put_element_variable_values(1, name, n + 1, pred[:, q, i])

    self.writer.SetInputData(multi_block)
    self.writer.Write()

  def convert_exodus_to_xml(self, exodus_file, xml_file):
    """Converts an Exodus II file to a VTK XML MultiBlock dataset."""

    # Read the Exodus file
    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(exodus_file)
    reader.Update()

    # Convert to a multi-block dataset
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(reader.GetNumberOfBlockArrays())

    for i in range(reader.GetNumberOfBlockArrays()):
        block_name = reader.GetBlockArrayName(i)
        block = reader.GetBlock(i)
        multiblock.SetBlock(i, block)
        multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), block_name)

    # Write the XML file
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(xml_file)
    writer.SetInputData(multiblock)
    writer.Write()

  def get_vtk_cell_type(self, exodus_element_type):
    """Map Exodus element types to VTK cell types."""
    if exodus_element_type == "QUAD":
        return vtk.VTK_QUAD
    elif exodus_element_type == "TRI":
        return vtk.VTK_TRIANGLE
    elif exodus_element_type == "TET":
        return vtk.VTK_TETRA
    elif exodus_element_type == "HEX":
        return vtk.VTK_HEXAHEDRON
    # elif exodus_element_type == "PYRAMID":
    #     return vtk.VTK_PYRAMID
    # elif exodus_element_type == "WEDGE":
    #     return vtk.VTK_WEDGE
    else:
        raise ValueError(f"Unsupported element type: {exodus_element_type}")

  def cell_type(self, vtk_cell_type, num_nodes_in_element):
    if vtk_cell_type == vtk.VTK_TETRA and num_nodes_in_element == 4:
      cell = vtk.vtkTetra()
    elif vtk_cell_type == vtk.VTK_HEXAHEDRON and num_nodes_in_element == 8:
      cell = vtk.vtkHexahedron()
    elif vtk_cell_type == vtk.VTK_QUAD and num_nodes_in_element == 4:
      cell = vtk.vtkQuad()
    elif vtk_cell_type == vtk.VTK_TRIANGLE and num_nodes_in_element == 3:
      cell = vtk.vtkTriangle()
    elif vtk_cell_type == vtk.VTK_PYRAMID and num_nodes_in_element == 5:
      cell = vtk.vtkPyramid()
    elif vtk_cell_type == vtk.VTK_WEDGE and num_nodes_in_element == 6:
      cell = vtk.vtkWedge()
    else:
      print(f"Skipping unsupported element type: {vtk_cell_type} with {num_nodes_in_element} nodes.")
      # continue
      cell = None
    return cell

# duck typing ... sort of?
class PostProcessor:
  def __init__(self, mesh_file: str, mesh_type = 'exodus') -> None:
    if mesh_type == 'exodus':
      self.pp = ExodusPostProcessor(mesh_file)
    elif mesh_type == 'vtk':
      self.pp = VtkPostProcessor(mesh_type)
    else:
      raise ValueError(f'Unsupported mesh type = {mesh_type}')
  
  def close(self):
    self.pp.close()

  def init(self, domain, output_file, node_variables, element_variables = []):
    self.pp.init(domain, output_file, node_variables, element_variables)

  def write_outputs(self, params, domain):
    self.pp.write_outputs(params, domain)
