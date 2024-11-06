from jax import numpy as jnp
from jax import vmap
from typing import List


def get_edges(domain, sset_names: List[str]):
  mesh = domain.fspace.mesh
  ssets = [mesh.sideSets[name] for name in sset_names]
  edges = jnp.vstack(ssets)
  edges = jnp.sort(edges, axis=1)
  edges = jnp.unique(edges, axis=0) # TODO not sure about this one
  return edges

def distance(x1, x2):
  return jnp.sqrt(
    jnp.power(x2[0] - x1[0], 2.) + 
    jnp.power(x2[1] - x1[1], 2.)
  )

def line_segment(x, segment):
  x1 = segment[:, 0]
  x2 = segment[:, 1]
  L = distance(x1, x2)
  x_c = (x1 + x2) / 2.

  f = (1. / L) * (
    (x[0] - x1[0]) * (x2[1] - x1[1]) - 
    (x[1] - x1[1]) * (x2[0] - x1[0])
  )
  t = (1. / L) * (
    jnp.power(L / 2., 2.) -
    jnp.power(distance(x, x_c), 2.)
  )
  varphi = jnp.sqrt(
    jnp.power(t, 2.0) +
    jnp.power(f, 4.0)
  )
  phi = jnp.sqrt(
    jnp.power(f, 2.0) + 
    (1. / 4.) * jnp.power(varphi - t, 2.)
  )
  return phi


def distance_function(domain, ssets, m=1.0):
  edges = get_edges(domain, ssets)
  m = 1.0
  segments = domain.fspace.mesh.coords[edges, :]
  def inner_func(x):
    Rs = vmap(line_segment, in_axes=(None, 0))(x, segments)
    Rs = vmap(lambda x: 1. / jnp.power(x, m))(Rs)
    return 1. / jnp.power(jnp.sum(Rs), 1. / m)
  return vmap(inner_func)
