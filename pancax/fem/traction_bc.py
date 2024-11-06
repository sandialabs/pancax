import jax
import jax.numpy as np

import pancax.fem.quadrature_rules as quadrature_rules
import pancax.fem.surface as surface

# def interpolate_nodal_field_on_edge(mesh, U, quadRule, edge):
#     fieldIndex = Surface.get_field_index(edge, mesh.conns)  
def interpolate_nodal_field_on_edge(conns, U, quadRule, edge):
    fieldIndex = surface.get_field_index(edge, conns)  
    nodalValues = surface.eval_field(U, fieldIndex)    
    return quadrature_rules.eval_at_iso_points(quadRule.xigauss, nodalValues)


# def compute_traction_potential_energy_on_edge(mesh, U, quadRule, edge, load, time):
#     uq = interpolate_nodal_field_on_edge(mesh, U, quadRule, edge)
#     Xq = interpolate_nodal_field_on_edge(mesh, mesh.coords, quadRule, edge)
#     edgeCoords = Surface.get_coords(mesh, edge)
def compute_traction_potential_energy_on_edge(coords, conns, U, quadRule, edge, load, time):
    uq = interpolate_nodal_field_on_edge(conns, U, quadRule, edge)
    Xq = interpolate_nodal_field_on_edge(conns, coords, quadRule, edge)
    edgeCoords = surface.get_coords(coords, conns, edge)
    edgeNormal = surface.compute_normal(edgeCoords)
    tq = jax.vmap(load, (0, None, None))(Xq, edgeNormal, time)
    integrand = jax.vmap(lambda u,t: u@t)(uq, tq)
    return -surface.integrate_values(quadRule, edgeCoords, integrand)


# def compute_traction_potential_energy(mesh, U, quadRule, edges, load, time=0.0):
#     return np.sum(jax.vmap(
#         compute_traction_potential_energy_on_edge, 
#         (None,None,None,0,None,None)
#     )(mesh, U, quadRule, edges, load, time) )

def compute_traction_potential_energy(coords, conns, U, quadRule, edges, load, time=0.0):
    return np.sum(jax.vmap(
        compute_traction_potential_energy_on_edge, 
        (None, None, None, None, 0, None, None)
    )(coords, conns, U, quadRule, edges, load, time))
