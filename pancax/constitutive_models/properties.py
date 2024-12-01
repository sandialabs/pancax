from jaxtyping import Array, Float
from typing import List, Union
import equinox as eqx
import jax


# TODO patch up error check in a good way
# probably just make a method to check type on other
class BoundedProperty(eqx.Module):
  prop_min: float = eqx.field(static=True)
  prop_max: float = eqx.field(static=True)
  prop_val: float
  # TODO
  # activation: Callable

  def __init__(
    self, 
    prop_min: float, 
    prop_max: float,
    key: jax.random.PRNGKey
  ) -> None:
    self.prop_min = prop_min
    self.prop_max = prop_max
    self.prop_val = jax.random.uniform(key, 1)

  def __call__(self):
    return self.prop_min + \
           (self.prop_max - self.prop_min) * jax.nn.sigmoid(self.prop_val)[0]

  def __add__(self, other):
    self._check_other_type(other, '+')
    return self.__call__() + other

  def __div__(self, other):
    self._check_other_type(other, '/')
    return self.__call__() / other

  def __mul__(self, other):
    self._check_other_type(other, '*')
    return self.__call__() * other

  def __sub__(self, other):
    self._check_other_type(other, '-')
    return self.__call__() - other

  def __radd__(self, other):
    self._check_other_type(other, '+')
    return other + self.__call__()

  def __rdiv__(self, other):
    self._check_other_type(other, '/')
    return other / self.__call__()

  def __rmul__(self, other):
    self._check_other_type(other, '*')
    return other * self.__call__()

  def __rsub__(self, other):
    self._check_other_type(other, '*')
    return other - self.__call__()

  def __rtruediv__(self, other):
    self._check_other_type(other, '/')
    return other / self.__call__()

  def __truediv__(self, other):
    self._check_other_type(other, '/')
    return self.__call__() / other

  def _check_other_type(self, other, op_str):
    if isinstance(other, Array) or \
       isinstance(other, float):
      pass
    else:
      raise TypeError(f'Unsupported type {type(other)} when doing {op_str} with BoundingProperty')


FixedProperty = float #= eqx.field(static=True)
Property = Union[BoundedProperty, FixedProperty]
