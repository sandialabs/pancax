from abc import abstractmethod

# from pancax.physics import incompressible_internal_force, internal_force
from typing import List

try:
    import vtk
except ModuleNotFoundError:
    print(
        "WARNING: Could not find vtk module. ."
        "You'll need to use another form of output"
    )

import jax
import jax.numpy as jnp
import os
import netCDF4 as nc
import numpy as onp

# import vtk


class BasePostProcessor:
    mesh_file: str = None
    node_variables: List[str] = None
    element_variables: List[str] = None

    # def __init__(
    #   self,
    #   mesh_file: str,
    #   node_variables: List[str],
    #   element_variables: List[str]
    # ) -> None:
    #   self.mesh_file = mesh_file
    #   self.node_variables = node_variables
    #   self.element_variables = element_variables

    def check_variable_names(self, domain, variables) -> None:
        for var in variables:
            if var == "internal_force" or \
               var == "incompressible_internal_force":
                continue

            if var not in domain.physics.var_name_to_method.keys():
                str = f"Unsupported variable requested for output {var}.\n"
                str += "Supported variables include:\n"
                for v in domain.physics.var_name_to_method.keys():
                    str += f"  {v}\n"
                raise ValueError(str)

    def get_node_variable_number(self, domain, variables) -> int:
        n = 0
        for var in variables:
            if var == "internal_force" or \
               var == "incompressible_internal_force":
                n = n + len(domain.physics.field_value_names)
            else:
                n = n + len(domain.physics.var_name_to_method[var]["names"])

        return n

    def get_element_variable_number(self, domain, variables) -> int:
        n = 0
        for var in variables:
            n = n + len(domain.physics.var_name_to_method[var]["names"])
        if n > 0:
            q_points = len(domain.fspace.quadrature_rule)
            return n * q_points
        else:
            return 0

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def init(
        self,
        domain,
        output_file: str,
        node_variables: List[str],
        element_variables: List[str],
    ) -> None:
        pass

    @abstractmethod
    def write_outputs(self, params, domain):
        pass


class ExodusPostProcessor(BasePostProcessor):
    def __init__(self, mesh_file: str) -> None:
        self.mesh_file = mesh_file
        self.output_file = None

    def close(self) -> None:
        pass

    def init(
        self,
        domain,
        output_file: str,
        node_variables: List[str],
        element_variables: List[str],
    ) -> None:
        self.output_file = output_file
        self.check_variable_names(domain, node_variables)
        self.check_variable_names(domain, element_variables)
        self.node_variables = node_variables
        self.element_variables = element_variables

        with nc.Dataset(self.mesh_file, "r") as src:
            with nc.Dataset(output_file, "w", format=src.data_model) as dst:
                # Copy global attributes from the
                # source file to the destination file
                dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

                # Copy dimensions from the source file to the destination file
                for name, dimension in src.dimensions.items():
                    dst.createDimension(
                        name, (len(dimension)
                               if not dimension.isunlimited()
                               else None)
                    )

                # Copy variables from the source file to the destination file
                for name, variable in src.variables.items():
                    # Create a new variable in the destination file
                    new_var = dst.createVariable(
                        name, variable.datatype, variable.dimensions
                    )
                    # Copy variable attributes
                    new_var.setncatts(
                        {k: variable.getncattr(k) for k in variable.ncattrs()}
                    )
                    # Copy variable data
                    new_var[:] = variable[:]

                # max_str_len = dst.dimensions['len_name']
                # TODO read from src file so things are consistent
                max_str_len = 256

                # now add new dimensions
                if len(node_variables) > 0:
                    # get total number of node variables
                    num_node_vars = 0
                    for var in node_variables:
                        for v in domain.physics.\
                                var_name_to_method[var]["names"]:
                            num_node_vars = num_node_vars + 1

                    dst.createDimension("num_nod_var", num_node_vars)
                    node_var_names = dst.createVariable(
                        "name_nod_var", "c", ("num_nod_var", "len_name")
                    )

                    n = 0
                    for var in node_variables:
                        for v in domain.\
                                physics.var_name_to_method[var]["names"]:
                            name = v.ljust(max_str_len)[:max_str_len]
                            # print(name)
                            node_var_names[n, :] = onp.array(list(name))
                            name = f"vals_nod_var{n + 1}"
                            new_var = dst.createVariable(
                                name, "double", ("time_step", "num_nodes")
                            )
                            n = n + 1

                if len(element_variables) > 0:
                    q_points = len(domain.fspace.quadrature_rule)
                    # get total number of node variables
                    num_elem_vars = 0
                    for var in element_variables:
                        for v in domain.\
                                physics.var_name_to_method[var]["names"]:
                            for _ in range(q_points):
                                num_elem_vars = num_elem_vars + 1

                    dst.createDimension("num_elem_var", num_elem_vars)
                    elem_var_names = dst.createVariable(
                        "name_elem_var", "c", ("num_elem_var", "len_name")
                    )

                    n = 0
                    for var in element_variables:
                        for v in domain.\
                                physics.var_name_to_method[var]["names"]:
                            for q in range(q_points):
                                name = f"{v}_{q + 1}"
                                name = name.ljust(max_str_len)[:max_str_len]
                                elem_var_names[n, :] = onp.array(list(name))
                                # NOTE this will only work for
                                # single block meshes
                                name = f"vals_elem_var{n + 1}eb1"
                                new_var = dst.createVariable(
                                    name, "double", ("time_step", "num_elem")
                                )
                                n = n + 1

    def write_outputs(self, params, problem):
        physics = problem.physics
        times = problem.times

        with nc.Dataset(self.output_file, "a") as dataset:
            ne = problem.domain.conns.shape[0]
            nq = len(problem.domain.fspace.quadrature_rule)

            def _vmap_func(n):
                return problem.physics.constitutive_model.\
                    initial_state()

            # TODO assumes constantly spaced timesteps
            dt = problem.times[1] - problem.times[0]
            state_old = jax.vmap(jax.vmap(_vmap_func))(
                jnp.zeros((ne, nq))
            )

            for n, time in enumerate(times):
                # write new time value
                time_var = dataset.variables["time_whole"]
                time_var[n] = time

                # useful for all methods later on
                us = physics.var_name_to_method["field_values"]["method"](
                    params, problem, time
                )
                # calculate something with state update at least once to update
                # state later
                _, state_new = physics.potential_energy(
                    params, problem.domain, time, us, state_old, dt
                )

                node_var_num = 0
                for var in self.node_variables:
                    if (
                        var == "internal_force"
                        or var == "incompressible_internal_force"
                    ):
                        assert False, "implement internal force stuff"
                        # us = jax.vmap(physics.\
                        # field_values, in_axes=(None, 0, None))(
                        #   params.fields, domain.coords, time)
                        # fs = onp.array(
                        # internal_force(domain, us, params.properties()))
                        # for i in range(fs.shape[1]):
                        #   self.exo.put_node_variable_values(
                        # f'internal_force_{self.index_to_component(i)}',
                        # n + 1, fs[:, i])
                    else:
                        output = physics.var_name_to_method[var]
                        pred = onp.array(
                            output["method"](params, problem, time)
                        )
                        if len(pred.shape) > 2:
                            for i in range(pred.shape[1]):
                                for j in range(pred.shape[2]):
                                    # k = pred.shape[1] * i + j
                                    assert False, "Support this"
                                    # self.exo.put_node_variable_values(
                                    # output['names'][k], n + 1, pred[:, i, j])
                        else:
                            for i in range(pred.shape[1]):
                                # self.exo.put_node_variable_values(
                                # output['names'][i], n + 1, pred[:, i])
                                # node_var = \
                                # `dataset.variables[output['names'][i]]
                                node_var = dataset.variables[
                                    f"vals_nod_var{node_var_num + 1}"
                                ]
                                node_var[n, :] = pred[:, i]
                                node_var_num = node_var_num + 1

                elem_var_num = 0
                for var in self.element_variables:
                    output = physics.var_name_to_method[var]
                    pred = onp.array(
                        output["method"](
                            params, problem, time, us, state_old, dt
                        )
                    )

                    # for q in range(pred.shape[1]):
                    # for i in range(pred.shape[2])
                    if len(pred.shape) == 2:
                        assert False, \
                            "Need to implement scalar element variable output"
                    elif len(pred.shape) == 3:
                        # this is the state variable case
                        for s in range(pred.shape[2]):
                            for q in range(pred.shape[1]):
                                elem_var = dataset.variables[
                                    # NOTE this will only work for
                                    # single block models
                                    f"vals_elem_var{elem_var_num + 1}eb1"
                                ]
                                elem_var[n, :] = pred[:, q, s]
                                elem_var_num = elem_var_num + 1
                    elif len(pred.shape) == 4:
                        temp = pred.reshape((pred.shape[0], pred.shape[1], 9))
                        for i in range(temp.shape[2]):
                            for q in range(pred.shape[1]):
                                elem_var = dataset.variables[
                                    # NOTE this will only work for
                                    # single block models
                                    f"vals_elem_var{elem_var_num + 1}eb1"
                                ]
                                elem_var[n, :] = temp[:, q, i]
                                elem_var_num = elem_var_num + 1
                    else:
                        assert False, f"Shape of output val is {pred.shape}"

                # finally update state
                state_old = state_new


class ExodusPostProcessor_old:
    def __init__(self, mesh_file: str) -> None:
        self.mesh_file = mesh_file
        self.exo = None
        self.node_variables = None
        self.element_variables = None

    def check_variable_names(self, domain, variables) -> None:
        for var in variables:
            if var == "internal_force" or \
               var == "incompressible_internal_force":
                continue

            if var not in domain.physics.var_name_to_method.keys():
                str = f"Unsupported variable requested for output {var}.\n"
                str += "Supported variables include:\n"
                for v in domain.physics.var_name_to_method.keys():
                    str += f"  {v}\n"
                raise ValueError(str)

    def close(self) -> None:
        self.exo.close()

    def copy_mesh(self, output_file: str) -> None:
        if os.path.isfile(output_file):
            os.remove(output_file)

        exo_temp = exodus.copy_mesh(self.mesh_file, output_file)
        exo_temp.close()
        self.exo = exodus.exodus(output_file, mode="a", array_type="numpy")

    def get_node_variable_number(self, domain, variables) -> int:
        n = 0
        for var in variables:
            if var == "internal_force" or \
               var == "incompressible_internal_force":
                n = n + len(domain.physics.field_value_names)
            else:
                n = n + len(domain.physics.var_name_to_method[var]["names"])

        return n

    def get_element_variable_number(self, domain, variables) -> int:
        n = 0
        for var in variables:
            n = n + len(domain.physics.var_name_to_method[var]["names"])
        if n > 0:
            q_points = len(domain.fspace.quadrature_rule)
            return n * q_points
        else:
            return 0

    def init(
        self,
        domain,
        output_file: str,
        node_variables: List[str],
        element_variables: List[str],
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
            if var == "internal_force" or \
               var == "incompressible_internal_force":
                self.exo.put_node_variable_name("internal_force_x", n)
                self.exo.put_node_variable_name("internal_force_y", n + 1)
                self.exo.put_node_variable_name("internal_force_z", n + 2)
                n = n + 3
            else:
                for v in domain.physics.var_name_to_method[var]["names"]:
                    self.exo.put_node_variable_name(v, n)
                    n = n + 1

        if len(element_variables) > 0:
            q_points = len(domain.fspace.quadrature_rule)
            n = 1
            for var in self.element_variables:
                for v in domain.physics.var_name_to_method[var]["names"]:
                    for q in range(q_points):
                        name = f"{v}_{q + 1}"
                        self.exo.put_element_variable_name(name, n)
                        n = n + 1

    def index_to_component(self, index):
        if index == 0:
            string = "x"
        elif index == 1:
            string = "y"
        elif index == 2:
            string = "z"
        else:
            raise ValueError("Should be 0, 1, or 2")
        return string

    def write_outputs(self, params, domain) -> None:
        physics = domain.physics
        times = domain.times
        for n, time in enumerate(times):
            self.exo.put_time(n + 1, time)

            for var in self.node_variables:
                if var == "internal_force" or \
                   var == "incompressible_internal_force":
                    us = jax.vmap(
                        physics.field_values, in_axes=(None, 0, None)
                    )(params.fields, domain.coords, time)
                    fs = onp.array(
                        internal_force(domain, us, params.properties())
                    )
                    for i in range(fs.shape[1]):
                        self.exo.put_node_variable_values(
                            f"internal_force_{self.index_to_component(i)}",
                            n + 1,
                            fs[:, i],
                        )
                else:
                    output = physics.var_name_to_method[var]
                    pred = onp.array(output["method"](params, domain, time))
                    if len(pred.shape) > 2:
                        for i in range(pred.shape[1]):
                            for j in range(pred.shape[2]):
                                k = pred.shape[1] * i + j
                                self.exo.put_node_variable_values(
                                    output["names"][k], n + 1, pred[:, i, j]
                                )
                    else:
                        for i in range(pred.shape[1]):
                            self.exo.put_node_variable_values(
                                output["names"][i], n + 1, pred[:, i]
                            )

            if len(self.element_variables) > 0:
                n_q_points = len(domain.fspace.quadrature_rule)
                for var in self.element_variables:
                    output = physics.var_name_to_method[var]
                    pred = onp.array(output["method"](params, domain, time))
                    for q in range(n_q_points):
                        for i in range(pred.shape[2]):
                            name = f'{output["names"][i]}_{q + 1}'
                            # NOTE this will only work on a single block
                            self.exo.put_element_variable_values(
                                1, name, n + 1, pred[:, q, i]
                            )


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
        domain,
        output_file: str,
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

                if var == "internal_force" or \
                   var == "incompressible_internal_force":
                    assert False
                    # us = jax.vmap(
                    #     physics.field_values, in_axes=(None, 0, None)
                    # )(params.fields, domain.coords, time)
                    # fs = onp.array(internal_force(
                    # domain, us, params.properties()))
                    # for i in range(fs.shape[1]):
                    #   self.exo.put_node_variable_values(
                    # f'internal_force_{self.index_to_component(i)}',
                    # n + 1, fs[:, i])
                else:
                    output = physics.var_name_to_method[var]
                    pred = onp.array(output["method"](params, domain, time))
                    if len(pred.shape) > 2:
                        assert False, "still need to hook this up"
                    #   for i in range(pred.shape[1]):
                    #     for j in range(pred.shape[2]):
                    #       k = pred.shape[1] * i + j
                    #       self.exo.put_node_variable_values(
                    # output['names'][k], n + 1, pred[:, i, j])
                    else:
                        # vector_array.SetNumberOfComponents(pred.shape[1])
                        vector_array.SetNumberOfComponents(3)

                        for i in range(pred.shape[0]):
                            if pred.shape[1] == 2:
                                vector_array.\
                                    InsertNextTuple(tuple(pred[i, :]) + (0.0,))
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
                        # time_array.InsertNextValue(n)
                        # Store the time step index or actual time value
                        multi_block.GetMetaData(n).Set("Time", time)

                        # multi_block.GetMetaData(n).Set(time_array)
                        # multi_block.GetMetaData(n).Set(
                        # vtk.vtkCompositeDataSet.NAME(), f'Time Step {n}')
                        # multi_block.GetMetaData(n)\
                        # .Set(vtk.vtkCompositeDataSet.TIME(), n)

                        #   self.exo.put_node_variable_values(
                        # output['names'][i], n + 1, pred[:, i])

            # if len(self.element_variables) > 0:
            #   n_q_points = len(domain.fspace.quadrature_rule)
            #   for var in self.element_variables:
            #     output = physics.var_name_to_method[var]
            #     pred = onp.array(output['method'](params, domain, time))
            #     for q in range(n_q_points):
            #       for i in range(pred.shape[2]):
            #         name = f'{output["names"][i]}_{q + 1}'
            #         # NOTE this will only work on a single block
            #         self.exo.put_element_variable_values(
            # 1, name, n + 1, pred[:, q, i])

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
            multiblock.GetMetaData(i).\
                Set(vtk.vtkCompositeDataSet.NAME(), block_name)

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
            raise ValueError(
                f"Unsupported element type: {exodus_element_type}"
            )

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
            print(
                f"Skipping unsupported element type: {vtk_cell_type} "
                f"with {num_nodes_in_element} nodes."
            )
            # continue
            cell = None
        return cell


# duck typing ... sort of?
class PostProcessor:
    def __init__(self, mesh_file: str, mesh_type="exodus") -> None:
        if mesh_type == "exodus":
            self.pp = ExodusPostProcessor(mesh_file)
        elif mesh_type == "vtk":
            self.pp = VtkPostProcessor(mesh_type)
        else:
            raise ValueError(f"Unsupported mesh type = {mesh_type}")

    def close(self):
        self.pp.close()

    def init(self, domain, output_file, node_variables, element_variables=[]):
        self.pp.init(domain, output_file, node_variables, element_variables)

    def write_outputs(self, params, domain):
        self.pp.write_outputs(params, domain)
