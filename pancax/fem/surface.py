# from optimism.JaxConfig import *
from jax import vmap
import jax.numpy as jnp

# this is hard coded for tri elements

def create_edges(coords, conns, edge_is_potentially_in_contact):
    edges = []
    for e,elem in enumerate(conns):
        edgeCoords = coords.take(elem,0)
        for n in range(3):
            nn = (n+1)%3
            edgeConn = jnp.array([n,nn])
            if (edge_is_potentially_in_contact(edgeCoords[edgeConn])):
                edges.append([e,n])
    return jnp.array(edges)


def get_coords(mesh, side):
    elem = mesh.conns[side[0],:]
    lSide = side[1]
    elemCoords = mesh.coords.take(elem,0)
    return elemCoords[ jnp.array([lSide, (lSide+1)%3]), : ]


def integrate_values(quadratureRule, coords, gaussField):
    _, wgauss = quadratureRule
    jac = jnp.linalg.norm(coords[0,:] - coords[1,:])
    dx = jac*wgauss
    return dx.dot(gaussField)


def integrate_function(quadratureRule, coords, field_func):
    xigauss, wgauss = quadratureRule
    jac = jnp.linalg.norm(coords[0,:] - coords[1,:])
    xgauss = jnp.array([coords[0] + (coords[1] - coords[0])*xi for xi in xigauss])
    dx = jac*wgauss
    return jnp.dot(vmap(field_func)(xgauss), dx)


def integrate_function_on_surface(quadratureRule, edges, mesh, func):
    F = vmap(integrate_function_on_edge, (None,0,None,None))
    return jnp.sum(F(quadratureRule, edges, mesh, func))


def integrate_function_on_edge(quadratureRule, edge, mesh, func):
    edgeCoords = get_coords(mesh, edge)
    xigauss, wgauss = quadratureRule
    jac = jnp.linalg.norm(edgeCoords[0,:] - edgeCoords[1,:])
    xgauss = edgeCoords[0] + jnp.outer(xigauss, edgeCoords[1] - edgeCoords[0])
    dx = jac*wgauss
    normal = compute_normal(edgeCoords)
    return jnp.dot(vmap(func, (0,None))(xgauss, normal), dx)


def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = jnp.array([tangent[1], -tangent[0]])
    return normal / jnp.linalg.norm(normal)


def get_field_index(edge, conns):
    elemIndex = edge[0]
    elemConns = conns[elemIndex]
    n1 = edge[1]
    n2 = (n1+1)%3
    return elemConns, jnp.array([n1,n2])


def eval_field(field, fieldIndex):
    return field[fieldIndex[0]][fieldIndex[1]]


def compute_edge_vectors(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = jnp.array([tangent[1], -tangent[0]])
    jac = jnp.linalg.norm(tangent)
    return tangent/jac, normal/jac, jac
