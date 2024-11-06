import equinox as eqx
import jax
import jax.numpy as jnp


class RBFBasis(eqx.Module):
  center: jax.Array
  sigma: jax.Array

  def __init__(
    self, 
    n_inputs: int, 
    n_neurons: int, 
    key: jax.random.PRNGKey
  ) -> None:
    ckey, skey  = jax.random.split(key)
    self.center = jax.random.normal(ckey, (n_neurons, n_inputs))
    self.sigma  = 1.0 * jnp.ones((n_neurons,), dtype=jnp.float64)

  def __call__(self, x):
    out = self.center - x
    out = jax.vmap(lambda a: jnp.dot(a, a))(out)
    out = jnp.exp(-out / jnp.square(self.sigma))
    return out

def rbf_normalization(hidden):
  return hidden / jnp.sum(hidden, axis=1)[:, None]
