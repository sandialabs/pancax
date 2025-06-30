import jax.numpy as np
from jax import jit, grad

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import QuadratureRule
from optimism.test.MeshFixture import MeshFixture
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism.material import HyperViscoelastic
# from optimism.ReadExodusMesh import read_exodus_mesh

# This example recreates the shear test considered in
#   Reese and Govindjee (1998). A theory of finite viscoelasticity and numerical aspects. 
#       https://doi.org/10.1016/S0020-7683(97)00217-5
# and in variational minimization form in 
#   Fancello, Ponthot and Stainier (2006). A variational formulation of constitutive models and updates in non-linear finite viscoelasticity.
#        https://doi.org/10.1002/nme.1525

class ShearTest(MeshFixture):

    def setUp(self):
        dummyDispGrad = np.eye(2)
        self.mesh = self.create_mesh_and_disp(21, 21, [0.,1.], [0.,1.],
                                                lambda x: dummyDispGrad.dot(x))[0]
        # self.mesh = read_exodus_mesh('./mesh/')
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=2, createNodeSetsFromSideSets=True)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        G_eq = 1.
        K_eq = 10*G_eq 
        G_neq_1 = 1.
        tau_1   = 10.
        props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat = HyperViscoelastic.create_material_model_functions(props)
        self.props = HyperViscoelastic.create_material_properties(props)

        self.EBCs = [
            FunctionSpace.EssentialBC(nodeSet='top', component=1),
            FunctionSpace.EssentialBC(nodeSet='top', component=0),
            FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
            FunctionSpace.EssentialBC(nodeSet='bottom', component=1)
        ]

        self.steps = 10
        self.dt = 1. / self.steps
        self.maxDisp = 1.0

    def run(self):
        mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.mat)
        dofManager = FunctionSpace.DofManager(self.fs, 2, self.EBCs)

        def create_field(Uu, disp):
            def get_ubcs(disp):
                V = np.zeros(self.mesh.coords.shape)
                index = (self.mesh.nodeSets['top'], 1)
                V = V.at[index].set(disp)
                return dofManager.get_bc_values(V)

            return dofManager.create_field(Uu, get_ubcs(disp))
        
        def energy_function_all_dofs(U, p):
            internalVariables = p.state_data
            return mechFuncs.compute_strain_energy(U, internalVariables, p.prop_data, self.dt)

        def compute_energy(Uu, p):
            U = create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, p)

        nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))
        integrate_dissipation = jit(mechFuncs.integrated_material_qoi)
        integrate_free_energy = jit(mechFuncs.compute_strain_energy)

        def write_output(Uu, p, step):
            from optimism import VTKWriter
            U = create_field(Uu, p.bc_data)
            plotName = 'mechanics-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displ', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            energyDensities, stresses = mechFuncs.\
                compute_output_energy_densities_and_stresses(U, p.state_data, p.prop_data, self.dt)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, energyDensities)
            cellStresses = FunctionSpace.project_quadrature_field_to_element_field(self.fs, stresses)
            dissipationDensities = mechFuncs.compute_output_material_qoi(U, p.state_data, p.prop_data, self.dt)
            cellDissipationDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, dissipationDensities)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.add_cell_field(name='piola_stress',
                                  cellData=cellStresses,
                                  fieldType=VTKWriter.VTKFieldType.TENSORS)
            writer.add_cell_field(name='dissipation_density',
                                  cellData=cellDissipationDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)

            for n in range(p.state_data.shape[-1]):
                state_temp = FunctionSpace.project_quadrature_field_to_element_field(self.fs, p.state_data[:, :, n])
                writer.add_cell_field(name=f'state_var_{n + 1}',
                                      cellData=state_temp,
                                      fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.write()

        Uu = dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs, prop_data=self.props)
        U = create_field(Uu, p.bc_data)
        self.objective = Objective.Objective(compute_energy, Uu, p)

        index = (self.mesh.nodeSets['top'], 1)

        time = 0.0
        times = [0.0]
        externalWorkStore = []
        dissipationStore = []
        incrementalPotentialStore = []
        forceHistory = [0.0]
        dispHistory = [0.0]

        coords = self.mesh.coords
        times_temp = np.zeros((coords.shape[0], 1))
        full_field_coords = [np.hstack((self.mesh.coords, times_temp))]
        full_field_disps = [U]

        write_output(Uu, p, 0)

        for step in range(1, self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            # disp = self.maxDisp * np.sin(2.0 * np.pi * self.freq * time)
            disp = self.maxDisp * step / (self.steps)

            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=True)
            U = create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, p.prop_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            write_output(Uu, p, step)
            
            force = np.array(nodal_forces(U, p).at[index].get())
            applied_disp = U.at[index].get()
            externalWorkStore.append( 0.5*np.tensordot((force + force_prev),(applied_disp - applied_disp_prev), axes=1) )
            incrementalPotentialStore.append(integrate_free_energy(U, ivs, p.prop_data, self.dt))

            forceHistory.append(np.sum(force))
            dispHistory.append(disp)

            dissipationStore.append(integrate_dissipation(U, ivs, p.prop_data, self.dt))

            times_temp = np.zeros((coords.shape[0], 1))
            full_field_coords.append(np.hstack((coords, times_temp)))
            full_field_disps.append(U)

            times.append(time)
            time += self.dt

        self.dt = 10.0 / self.steps

        # relax step
        for step in range(self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=True)
            U = create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, p.prop_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            write_output(Uu, p, step + self.steps)
            
            force = np.array(nodal_forces(U, p).at[index].get())
            applied_disp = U.at[index].get()
            externalWorkStore.append( 0.5*np.tensordot((force + force_prev),(applied_disp - applied_disp_prev), axes=1) )
            incrementalPotentialStore.append(integrate_free_energy(U, ivs, p.prop_data, self.dt))

            forceHistory.append(np.sum(force))
            dispHistory.append(disp)

            dissipationStore.append(integrate_dissipation(U, ivs, p.prop_data, self.dt))

            times_temp = np.zeros((coords.shape[0], 1))
            full_field_coords.append(np.hstack((coords, times_temp)))
            full_field_disps.append(U)

            times.append(time)
            time += self.dt

        # storing for plots
        with open("energy_histories.npz",'wb') as f:
            np.savez(f, externalWork=np.cumsum(np.array(externalWorkStore)), dissipation=np.cumsum(np.array(dissipationStore)), algorithmicPotential=np.array(incrementalPotentialStore), time=np.array(times))

        with open("force_disp_histories.npz",'wb') as f:
            np.savez(f, forces=np.array(forceHistory), disps=np.array(dispHistory))


        # Converting stuff to csv via pandas
        import numpy as onp
        import pandas as pd

        # output global data
        data = onp.load("force_disp_histories.npz")

        # Check all arrays are 1D and same length
        arrays = {key: data[key] for key in data.files}
        lengths = [arr.shape[0] for arr in arrays.values()]
        if not all(arr.ndim == 1 for arr in arrays.values()):
            raise ValueError("Not all arrays are 1D")
        if len(set(lengths)) > 1:
            raise ValueError("Arrays are not the same length")

        # Combine into one DataFrame
        df = pd.DataFrame(arrays)
        df['times'] = times
        df.to_csv("global_data.csv", index=False)

        # output full field data

        full_field_coords = onp.array(np.vstack(full_field_coords))
        full_field_disps = onp.array(np.vstack(full_field_disps))

        d = {
            'x': full_field_coords[:, 0],
            'y': full_field_coords[:, 1],
            't': full_field_coords[:, 2],
            'u_x': full_field_disps[:, 0],
            'u_y': full_field_disps[:, 1]
        }
        df = pd.DataFrame(d)
        df.to_csv('full_field_data.csv', index=False)



app = ShearTest()
app.setUp()
app.run()