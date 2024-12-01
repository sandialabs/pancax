from .base import Optimizer
from typing import Callable
from typing import Optional
import equinox as eqx
import optax


class Adam(Optimizer):
  def __init__(
    self,
    loss_function: Callable,
    has_aux: Optional[bool] = False,
    jit: Optional[bool] = True,
    learning_rate: Optional[float] = 1.0e-3,
    transition_steps: Optional[int] = 500,
    decay_rate: Optional[float] = 0.99,
    clip_gradients: Optional[bool] = False,
    filter_spec: Optional[Callable] = None
  ) -> None:
    super().__init__(loss_function, has_aux, jit)
    scheduler = optax.exponential_decay(
      init_value=learning_rate,
      transition_steps=transition_steps,
      decay_rate=decay_rate
    )
    if clip_gradients:
      self.opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
      )
    else:
      self.opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
      )
    
    if filter_spec is None:
      self.loss_and_grads = eqx.filter_value_and_grad(self.loss_function, has_aux=self.has_aux)
    else:
      self.loss_and_grads = eqx.filter_value_and_grad(self.loss_function.filtered_loss, has_aux=self.has_aux)
    self.filter_spec = filter_spec
  
  def make_step_method(self):
    if self.filter_spec is None:
      def step(params, domain, opt_st):
        loss, grads = self.loss_and_grads(params, domain)
        updates, opt_st = self.opt.update(grads, opt_st)
        params = eqx.apply_updates(params, updates)
        # add grad props to output
        # TODO what to do about below?
        # loss[1].update({'dprops': grads.properties.prop_params})
        return params, opt_st, loss
    else:
      def step(params, domain, opt_st):
        diff_params, static_params = eqx.partition(params, self.filter_spec)
        loss, grads = self.loss_and_grads(diff_params, static_params, domain)
        updates, opt_st = self.opt.update(grads, opt_st)
        params = eqx.apply_updates(params, updates)

        # add grad props to output
        # loss[1].update({'dprops': grads.properties()})
        return params, opt_st, loss

    return step
