from jax._src.random import PRNGKey as PRNGKey
from jax import random
from jaxtyping import Array, Float
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp



class Properties(eqx.Module):
  """
  :param prop_mins: Minimum allowable properties
  :param prop_maxs: Maximum allowable properties
  :param prop_params: Actual tunable parameters
  """
  prop_mins: jax.Array = eqx.field(static=True)
  prop_maxs: jax.Array = eqx.field(static=True)
  prop_params: jax.Array
  activation_func: Callable

  def __init__(
    self, 
    prop_mins: jax.Array,
    prop_maxs: jax.Array,
    key: jax.random.PRNGKey,
    activation_func: Optional[Callable] = jax.nn.sigmoid
  ) -> None:
    """
    :param prop_mins: Minimum allowable properties
    :param prop_maxs: Maximum allowable properties
    :param key: rng key
    """

    self.prop_mins = jnp.array(prop_mins)
    self.prop_maxs = jnp.array(prop_maxs)
    self.prop_params = jax.random.uniform(key, self.prop_mins.shape)
    self.activation_func = activation_func

  def __call__(self) -> Float[Array, "np"]:
    """
    :return: Predicted properties
    """
    props = self.prop_mins + \
            (self.prop_maxs - self.prop_mins) * self.activation_func(self.prop_params)
    return props


class FixedProperties(Properties):
  def __init__(self, props: jax.Array) -> None:
    """
    :param props: Property values to be fixed
    """
    key = random.key(0)
    super().__init__(props, props, key)
