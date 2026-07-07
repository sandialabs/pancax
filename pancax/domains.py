from jaxtyping import Array, Float, Int
from pancax.fem import DofManager, FunctionSpace, Mesh, SimplexTriElement, \
    NonAllocatedFunctionSpace, QuadratureRule
from pancax.fem import create_higher_order_mesh_from_simplex_mesh
from pancax.fem import read_exodus_mesh
from pancax.physics_kernels import LaplaceBeltrami
from pancax.post_processor import PostProcessor
from pancax.timer import Timer
from scipy.sparse import linalg
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp
import netCDF4 as nc
import numpy as onp


class SimulationTimesNotUniqueException(Exception):
    pass


class SimulationTimesNotStrictlyIncreasingException(Exception):
    pass


class BaseDomain(eqx.Module):
    mesh_file: str
    mesh: Mesh
    coords: Float[Array, "nn nd"]
    times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
    dof_manager: DofManager

    def __init__(
        self, mesh_file: str, times: Float[Array, "nt"],
        p_order: Optional[int] = 1
    ) -> None:
        with Timer("Reading Mesh..."):
            mesh = read_exodus_mesh(mesh_file)
            # if tri mesh, we can make it higher order from lower order
            if type(mesh.parentElement) is SimplexTriElement:
                mesh = create_higher_order_mesh_from_simplex_mesh(
                    mesh, p_order, copyNodeSets=True
                )
            else:
                print(
                    "WARNING: Ignoring polynomial \
                    order flag for non tri mesh"
                )

            # checking provided simulation times are unique
            if len(times) != len(set(times.tolist())):
                raise SimulationTimesNotUniqueException()

            # checking provided times are strictly increasing
            for i in range(len(times) - 1):
                if times[i] >= times[i + 1]:
                    raise SimulationTimesNotStrictlyIncreasingException()

            self.mesh_file = mesh_file
            self.mesh = mesh
            self.coords = jnp.array(mesh.coords)
            self.times = times
            self.dof_manager = None

    def update_dof_manager(self, dirichlet_bcs, n_dofs):
        dof_manager = DofManager(self.mesh, n_dofs, dirichlet_bcs)
        dof_manager.isUnknown = jnp.array(
            dof_manager.isUnknown, dtype=jnp.bool
        )
        dof_manager.unknownIndices = jnp.array(dof_manager.unknownIndices)
        new_pytree = eqx.tree_at(lambda x: x.dof_manager, self, dof_manager)
        return new_pytree


class CollocationDomain(BaseDomain):
    mesh_file: str
    mesh: Mesh
    coords: Float[Array, "nn nd"]
    times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
    dof_manager: DofManager

    def __init__(
        self, mesh_file: str, times: Float[Array, "nt"],
        p_order: Optional[int] = 1
    ) -> None:
        super().__init__(mesh_file, times, p_order=p_order)


class CollocationDataLoader(eqx.Module):
    indices: onp.ndarray
    inputs: Float[Array, "bs ni"]
    outputs: Float[Array, "bs no"]

    def __init__(
        self,
        domain: CollocationDomain,
        num_fields: int
    ) -> None:
        inputs = []

        # For now, just a simple collection of mesh coordinates
        # TODO add sampling strategies
        coords = domain.coords
        ones = jnp.ones((coords.shape[0], 1))
        for time in domain.times:
            times = time * ones
            temp = jnp.hstack((coords, times))
            inputs.append(temp)

        inputs = jnp.vstack(inputs)
        outputs = jnp.zeros((inputs.shape[0], num_fields))

        indices = onp.arange(len(inputs))

        self.indices = indices
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def dataloader(self, batch_size: int):
        perm = onp.random.permutation(self.indices)
        start = 0
        end = batch_size
        while end <= len(self):
            batch_perm = perm[start:end]
            yield self.inputs[batch_perm], self.outputs[batch_perm]
            start = end
            end = start + batch_size


class VariationalDomain(BaseDomain):
    mesh_file: str
    mesh: Mesh
    coords: Float[Array, "nn nd"]
    times: Union[Float[Array, "nt"], Float[Array, "nn 1"]]
    conns: Int[Array, "ne nnpe"]
    dof_manager: DofManager
    fspace: FunctionSpace
    fspace_centroid: FunctionSpace

    def __init__(
        self,
        mesh_file: str,
        times: Float[Array, "nt"],
        p_order: Optional[int] = 1,
        q_order: Optional[int] = 2,
    ):
        super().__init__(mesh_file, times, p_order=p_order)
        self.conns = jnp.array(self.mesh.conns)
        self.fspace = NonAllocatedFunctionSpace(
            self.mesh, QuadratureRule(self.mesh.parentElement, q_order)
        )
        self.fspace_centroid = NonAllocatedFunctionSpace(
            self.mesh, QuadratureRule(self.mesh.parentElement, 1)
        )


class DeltaPINNDomain(VariationalDomain):
    n_eigen_values: int
    physics: LaplaceBeltrami
    eigen_modes: Float[Array, "nn nev"]

    def __init__(
        self,
        mesh_file: str,
        times: Float[Array, "nt"],
        n_eigen_values: int,
        p_order: Optional[int] = 1,
        q_order: Optional[int] = 2,
    ):
        super().__init__(mesh_file, times, p_order=p_order, q_order=q_order)
        self.n_eigen_values = n_eigen_values
        physics = LaplaceBeltrami()
        # physics = physics.update_normalization(self)
        self.physics = physics.update_var_name_to_method()
        self.eigen_modes = self.solve_eigen_problem()

    def solve_eigen_problem(self):
        # physics = LaplaceBeltrami()
        dof_manager = DofManager(self.mesh, 1, [])
        Uu = jnp.zeros(dof_manager.get_unknown_size())
        U = dof_manager.create_field(Uu)
        ne = self.conns.shape[0]
        nq = len(self.fspace.quadrature_rule)
        state_old = jnp.zeros((ne, nq, 0))
        dt = 0.
        # TODO need to define these methods
        K = self.physics.stiffness_matrix(
            (), self, 0.0, U, state_old, dt, dof_manager
        )
        M = self.physics.mass_matrix(
            (), self, 0.0, U, state_old, dt, dof_manager
        )

        with Timer("eigen solve"):
            nModes = self.n_eigen_values
            mu, modes = linalg.eigsh(A=K, k=nModes, M=M, which="SM")
            # mu, modes = linalg.eigsh(A=M, k=nModes, M=K, which="LA")

            lambdas = 1.0 / mu
            for n in range(len(lambdas)):
                print(f"  Eigen mode {n} = {1. / lambdas[n]}")

        # dummy params
        class DummyParams:
            is_ensemble = False
            n_ensemble = 1

        params = DummyParams()

        with Timer("post-processing"):
            pp = PostProcessor(self.mesh_file)
            pp.init(
                params, self, "output-eigen.e", node_variables=["field_values"]
            )

            with nc.Dataset(pp.pp.output_file, "a") as out:
                for n in range(len(lambdas)):
                    print(f"  Post-processing eigenvector {n}")
                    field = dof_manager.create_field(modes[:, n])
                    field = onp.asarray(field)

                    time_var = out.variables["time_whole"]
                    time_var[n] = 1.0 / lambdas[n]

                    node_var = out.variables["vals_nod_var1"]
                    node_var[n, :] = field[:, 0]

                # TODO will fail on no exodus post-processor
                # pp.pp.exo.put_time(n + 1, 1. / lambdas[n])
                # pp.pp.exo.put_node_variable_values('u', n + 1, field[:, 0])

            pp.close()

        # normalizing modes by default
        modes = jnp.array(modes[:, 0:len(lambdas) - 1])
        print(modes.shape)
        modes = (modes - jnp.min(modes, axis=0)) / (
            jnp.max(modes, axis=0) - jnp.min(modes, axis=0)
        )

        return modes
