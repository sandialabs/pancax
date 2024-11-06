# from jax import vmap
# from pancax import fem
# from pancax.fem import quadrature_rules
# from pancax.fem import surface
# from pancax.fem.surface import create_edges
# import jax.numpy as np

# Nx = 4
# Ny = 4
# L = 1.2
# W = 1.5
# xRange = [0, L]
# yRange = [0, W]
# targetDispGrad = np.zeros((2, 2))

# def create_mesh_and_disp(Nx, Ny, xRange, yRange, initial_disp_func, setNamePostFix=''):
#     coords, conns = fem.create_structured_mesh_data(Nx, Ny, xRange, yRange)
#     tol = 1e-8
#     nodeSets = {}
#     nodeSets['left'+setNamePostFix] = np.flatnonzero(coords[:,0] < xRange[0] + tol)
#     nodeSets['bottom'+setNamePostFix] = np.flatnonzero(coords[:,1] < yRange[0] + tol)
#     nodeSets['right'+setNamePostFix] = np.flatnonzero(coords[:,0] > xRange[1] - tol)
#     nodeSets['top'+setNamePostFix] = np.flatnonzero(coords[:,1] > yRange[1] - tol)
#     nodeSets['all_boundary'+setNamePostFix] = np.flatnonzero(
#         (coords[:,0] < xRange[0] + tol) |
#         (coords[:,1] < yRange[0] + tol) |
#         (coords[:,0] > xRange[1] - tol) |
#         (coords[:,1] > yRange[1] - tol) 
#     )
    
#     def is_edge_on_left(xyOnEdge):
#         return np.all( xyOnEdge[:,0] < xRange[0] + tol  )

#     def is_edge_on_bottom(xyOnEdge):
#         return np.all( xyOnEdge[:,1] < yRange[0] + tol  )

#     def is_edge_on_right(xyOnEdge):
#         return np.all( xyOnEdge[:,0] > xRange[1] - tol  )
    
#     def is_edge_on_top(xyOnEdge):
#         return np.all( xyOnEdge[:,1] > yRange[1] - tol  )

#     sideSets = {}
#     sideSets['left'+setNamePostFix] = create_edges(coords, conns, is_edge_on_left)
#     sideSets['bottom'+setNamePostFix] = create_edges(coords, conns, is_edge_on_bottom)
#     sideSets['right'+setNamePostFix] = create_edges(coords, conns, is_edge_on_right)
#     sideSets['top'+setNamePostFix] = create_edges(coords, conns, is_edge_on_top)
    
#     allBoundaryEdges = np.vstack([s for s in sideSets.values()])
#     sideSets['all_boundary'+setNamePostFix] = allBoundaryEdges

#     blocks = {'block'+setNamePostFix: np.arange(conns.shape[0])}
#     mesh = fem.construct_mesh_from_basic_data(coords, conns, blocks, nodeSets, sideSets)
#     return mesh, vmap(initial_disp_func)(mesh.coords)

# mesh, U = create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x: np.dot(targetDispGrad, x))
# quadRule = quadrature_rules.create_quadrature_rule_1D(degree=2)


# def test_integrate_perimeter():
#     p = surface.integrate_function_on_surface(quadRule,
#                                               mesh.sideSets['all_boundary'],
#                                               mesh,
#                                               lambda x, n: 1.0)
#     assert np.abs(p - 2 * (L + W)) < 1e-14

    
# def test_integrate_quadratic_fn_on_surface():
#     I = surface.integrate_function_on_surface(quadRule,
#                                               mesh.sideSets['top'],
#                                               mesh,
#                                               lambda x, n: x[0]**2)
#     assert np.abs(I - L**3 / 3.) < 1e-14
    
# def test_integrate_function_on_surface_that_uses_coords_and_normal():
#     I = surface.integrate_function_on_surface(quadRule,
#                                               mesh.sideSets['all_boundary'],
#                                               mesh,
#                                               lambda x, n: np.dot(x,n))
#     assert np.abs(I - 2 * L * W) < 1e-14
