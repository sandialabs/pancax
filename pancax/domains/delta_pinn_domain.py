from jaxtyping import Array, Float
from pancax.fem import DofManager
from pancax.physics_kernels import LaplaceBeltrami
from pancax.post_processor import PostProcessor
from pancax.timer import Timer
from scipy.sparse import linalg
from typing import Optional
from .variational_domain import VariationalDomain
import jax.numpy as jnp
import netCDF4 as nc
import numpy as onp


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
    q_order: Optional[int] = 2
  ):
    super().__init__(mesh_file, times, p_order=p_order, q_order=q_order)
    self.n_eigen_values = n_eigen_values
    physics = LaplaceBeltrami()
    physics = physics.update_normalization(self)
    self.physics = physics.update_var_name_to_method()
    self.eigen_modes = self.solve_eigen_problem()

  # def __pos

  def solve_eigen_problem(self):
    # physics = LaplaceBeltrami()
    dof_manager = DofManager(self.mesh, 1, [])
    Uu = jnp.zeros(dof_manager.get_unknown_size())
    U = dof_manager.create_field(Uu)
    # TODO need to define these methods
    K = self.physics.stiffness_matrix((), self, 0., U, dof_manager)
    M = self.physics.mass_matrix((), self, 0., U, dof_manager)

    with Timer('eigen solve'):
      nModes = self.n_eigen_values
      mu, modes = linalg.eigsh(
        A=K, k=nModes, M=M, which='SM'
      )

      lambdas = 1. / mu
      for n in range(len(lambdas)):
        print(f'  Eigen mode {n} = {1. / lambdas[n]}')

    with Timer('post-processing'):
      pp = PostProcessor(self.mesh_file)
      pp.init(self, 'output-eigen.e',
        node_variables=[
          'field_values'
        ]        
      )

      with nc.Dataset(pp.pp.output_file, 'a') as out:
        for n in range(len(lambdas)):
          print(f'  Post-processing eigenvector {n}')
          field = dof_manager.create_field(modes[:, n])
          field = onp.asarray(field)

          time_var = out.variables['time_whole']
          time_var[n] = 1. / lambdas[n]

          node_var = out.variables[f'vals_nod_var1']
          node_var[n, :] = field[:, 0]

        # TODO will fail on no exodus post-processor
        # pp.pp.exo.put_time(n + 1, 1. / lambdas[n])
        # pp.pp.exo.put_node_variable_values('u', n + 1, field[:, 0])
        
      pp.close()

    # normalizing modes by default
    modes = jnp.array(modes[:, 0:len(lambdas) - 1])
    print(modes.shape)
    modes = (modes - jnp.min(modes, axis=0)) / \
            (jnp.max(modes, axis=0) - jnp.min(modes, axis=0))

    return modes
