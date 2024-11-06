from copy import copy
from copy import deepcopy
from pancax.fem import Mesh
import jax
import jax.numpy as jnp



# def make_mesh_time_dependent(mesh, n_time_steps):
def make_mesh_time_dependent(mesh, times):
  n_time_steps = len(times)
  
  # some constants
  n_elements = mesh.conns.shape[0]
  n_nodes = mesh.coords.shape[0]

  # arrays to return in a new mesh object
  coords = copy(mesh.coords)
  conns = copy(mesh.conns)
  simplexNodesOrdinals = copy(mesh.simplexNodesOrdinals)
  blocks = deepcopy(mesh.blocks)
  nodeSets = deepcopy(mesh.nodeSets)
  sideSets = deepcopy(mesh.sideSets)

  for n in range(1, n_time_steps):
    coords = jnp.vstack((coords, mesh.coords))
    conns = jnp.vstack((conns, mesh.conns + n * n_nodes))
    simplexNodeOrdinals = jnp.vstack((simplexNodesOrdinals, mesh.simplexNodesOrdinals))

    for key in mesh.blocks.keys():
      blocks[key] = jnp.hstack((blocks[key], mesh.blocks[key] + n * n_elements))

    for key in mesh.nodeSets.keys():
      nodeSets[key] = jnp.hstack((nodeSets[key], mesh.nodeSets[key] + n * n_nodes))

    for key in mesh.sideSets.keys():
      sideSets[key] = jnp.hstack((sideSets[key], mesh.sideSets[key] + n * n_elements))

  return Mesh(
    coords=coords, conns=conns, simplexNodesOrdinals=simplexNodesOrdinals,
    parentElement=mesh.parentElement, parentElement1d=mesh.parentElement1d,
    blocks=blocks, nodeSets=nodeSets, sideSets=sideSets
  )

def make_times_column(n_nodes: int, times_in: jax.Array):
  times = jnp.zeros(n_nodes * len(times_in), dtype=jnp.float64)
  for n, time in enumerate(times_in):
    times = times.at[n * n_nodes:(n + 1) * n_nodes].set(time * jnp.ones(n_nodes, dtype=jnp.float64))
  return times
