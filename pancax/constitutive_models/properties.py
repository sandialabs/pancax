from jaxtyping import Array, Float
from typing import Union
import equinox as eqx
import jax
import jax.numpy as jnp


# TODO patch up error check in a good way
# probably just make a method to check type on other
class BoundedProperty(eqx.Module):
    prop_min: float
    prop_max: float
    prop_val: Float[Array, "n"]
    # TODO
    # activation: Callable

    def __init__(
        self, prop_min: float, prop_max: float, key: jax.random.PRNGKey
    ) -> None:
        self.prop_min = prop_min
        self.prop_max = prop_max

        if len(key.shape) == 1:
            self.prop_val = self._sample(key, prop_min, prop_max)
        elif len(key.shape) == 2:
            @eqx.filter_vmap
            def vmap_func(key):
                return self._sample(key, prop_min, prop_max)
            self.prop_val = vmap_func(key)
        else:
            assert False, f"This shouldn't happen key = {key}"

    def __call__(self):
        return (
            self.prop_min +
            (self.prop_max - self.prop_min) *
            jax.nn.sigmoid(self.prop_val)[0]
        )

    def __repr__(self):
        # return str(self.__call__())
        return str((
            self.prop_min +
            (self.prop_max - self.prop_min) *
            jax.nn.sigmoid(self.prop_val)
        ))

    def __add__(self, other):
        self._check_other_type(other, "+")
        return self.__call__() + other

    def __div__(self, other):
        self._check_other_type(other, "/")
        return self.__call__() / other

    def __mul__(self, other):
        self._check_other_type(other, "*")
        return self.__call__() * other

    def __sub__(self, other):
        self._check_other_type(other, "-")
        return self.__call__() - other

    def __radd__(self, other):
        self._check_other_type(other, "+")
        return other + self.__call__()

    def __rdiv__(self, other):
        self._check_other_type(other, "/")
        return other / self.__call__()

    def __rmul__(self, other):
        self._check_other_type(other, "*")
        return other * self.__call__()

    def __rsub__(self, other):
        self._check_other_type(other, "*")
        return other - self.__call__()

    def __rtruediv__(self, other):
        self._check_other_type(other, "/")
        return other / self.__call__()

    def __truediv__(self, other):
        self._check_other_type(other, "/")
        return self.__call__() / other

    def _check_other_type(self, other, op_str):
        if isinstance(other, Array) or isinstance(other, float):
            pass
        else:
            raise TypeError(
                f"Unsupported type {type(other)} when doing \
                {op_str} with BoundingProperty"
            )

    def _sample(self, key, lb, ub):
        if lb == ub:
            return jnp.zeros(1)

        p_actual = jax.random.uniform(key, 1, minval=lb, maxval=ub)
        y = (p_actual - lb) / (ub - lb)
        return jnp.log(y / (1. - y))


FixedProperty = float
Property = Union[BoundedProperty, FixedProperty]
