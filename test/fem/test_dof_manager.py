import pytest

Nx = 4
Ny = 5
nNodes = Nx * Ny
nFields = 2
nDof = nFields * nNodes


@pytest.fixture
def dof_manager_test_fixture():
    from pancax.bcs import DirichletBC
    from pancax.fem import DofManager
    from .utils import create_mesh_and_disp
    import jax.numpy as jnp

    xRange = [0.0, 1.0]
    yRange = [0.0, 1.0]

    mesh, _ = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: 0 * x)

    ebcs = [
        DirichletBC(nodeSet="top", component=0),
        DirichletBC(nodeSet="right", component=1),
    ]

    dofManager = DofManager(mesh, nFields, ebcs)

    U = jnp.zeros((nNodes, nFields))
    U = U.at[:, 1].set(1.0)
    U = U.at[mesh.nodeSets["top"], 0].set(2.0)
    U = U.at[mesh.nodeSets["right"], 1].set(3.0)

    return dofManager, U


def test_get_bc_size(dof_manager_test_fixture):
    dofManager, _ = dof_manager_test_fixture
    # number of dofs from top, field 0
    nEbcs = Nx
    # number of dofs from right, field 1
    nEbcs += Ny
    assert dofManager.get_bc_size() == nEbcs


def test_get_unknown_size(dof_manager_test_fixture):
    dofManager, _ = dof_manager_test_fixture
    # number of dofs from top, field 0
    nEbcs = Nx
    # number of dofs from right, field 1
    nEbcs += Ny
    assert dofManager.get_unknown_size() == nDof - nEbcs


def test_slice_unknowns_with_dof_indices(dof_manager_test_fixture):
    import jax.numpy as jnp

    dofManager, U = dof_manager_test_fixture

    Uu = dofManager.get_unknown_values(U)
    Uu_x = dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None), 0))
    jnp.array_equal(Uu_x, jnp.zeros(Nx * (Ny - 1)))
    Uu_y = dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None), 1))
    jnp.array_equal(Uu_y, jnp.ones(Ny * (Nx - 1)))


def test_create_field_and_get_bc_values(dof_manager_test_fixture):
    import jax.numpy as jnp

    dofManager, U = dof_manager_test_fixture

    Uu = jnp.zeros(dofManager.get_unknown_size())
    U_new = dofManager.create_field(Uu, Ubc=5.0)
    assert U_new.shape == U.shape
    assert jnp.allclose(dofManager.get_bc_values(U_new), 5.0)
