from .base import Optimizer
from optax import tree_utils as otu
# from optimistix._misc import filter_cond
from typing import Any, Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def value_and_grad_from_state(value_fn):
  def _value_and_grad(
    params, domain,
    *fn_args: Any,
    state,
    **fn_kwargs: dict[str, Any],
  ):
    value = otu.tree_get(state, 'value')
    grad = otu.tree_get(state, 'grad')
    aux = dict(energy=jnp.array(0.0), residual=jnp.array(1000.0))
    if (value is None) or (grad is None):
      raise ValueError(
        'Value or gradient not found in the state. '
        'Make sure that these values are stored in the state by the '
        'optimizer.'
      )
    # (value, aux), grad = filter_cond(
    #     (~jnp.isinf(value)) & (~jnp.isnan(value)),
    #     lambda p, a, kwa: ((value, aux), grad),
    #     lambda p, a, kwa: eqx.filter_value_and_grad(value_fn, has_aux=True)(p, domain, *a, **kwa),
    #     params,
    #     fn_args,
    #     fn_kwargs,
    # )
    print(value)
    (value, aux), grad = filter_cond(
      (~jnp.isinf(value)) & (~jnp.isnan(value)),
      lambda p: ((value, aux), grad),
      # lambda p: (value, grad),
      lambda p: eqx.filter_value_and_grad(value_fn, has_aux=True)(p, domain),
      params
    )
    # TODO
    # probably really slow right now
    # value, grad = eqx.filter_value_and_grad(value_fn, has_aux=True)(params, domain)
    return (value, aux), grad

  return _value_and_grad


class LBFGS(Optimizer):
  def __init__(
    self, 
    loss_function: Callable, 
    has_aux: Optional[bool] = False, 
    jit: Optional[bool] = False,
    learning_rate: Optional[float] = 1.0e-1
  ) -> None:
    super().__init__(loss_function, has_aux, jit)

    # TODO setup up optimizer stuff here
    linesearch = optax.scale_by_backtracking_linesearch(
      max_backtracking_steps=15, store_grad=True
    )
    # self.opt = optax.chain(

    # )
    self.opt = optax.lbfgs(
      learning_rate=learning_rate,
      linesearch=linesearch
    )

    # self.loss_and_grads = eqx.filter_value_and_grad(self.loss_function, has_aux=self.has_aux)
    self.loss_and_grads = value_and_grad_from_state(self.loss_function)

  def make_step_method(self):
    def step(params, domain, opt_st):
      loss, grads = self.loss_and_grads(params, domain, state=opt_st)
      updates, opt_st = self.opt.update(
        grads, opt_st, params, value=loss, grad=grads, 
        value_fn=self.loss_function
      )
      print('here')
      print(updates)
      print('here')
      print(opt_st)
      print('here')
      params = eqx.apply_updates(params, updates)
      return params, opt_st, loss

    return step
