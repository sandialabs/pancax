from .base import AbstractOptimizer
from typing import Callable, Optional, Union
import equinox as eqx
import optax


class Adam(AbstractOptimizer):
    epoch: int
    has_aux: bool
    jit: bool
    loss_function: Callable  # TODO further type me
    loss_and_grads: Callable
    step: Union[None, Callable]
    #
    opt: any

    def __init__(
        self,
        loss_function: Callable,
        learning_rate: float,
        # TODO this option should probably be on the base class
        clip_gradients: Optional[bool] = False,
        decay_rate: Optional[float] = 0.99,
        has_aux: Optional[bool] = False,
        jit: Optional[bool] = True,
        transition_steps: Optional[int] = 500
    ) -> None:
        super().__init__(
            loss_function,
            has_aux=has_aux,
            jit=jit
        )

        # TODO figure out how best to handle scheduler
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )

        if clip_gradients:
            self.opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )
        else:
            self.opt = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )

    def make_step_method(self, filter_spec: Callable) -> Callable:
        def step(params, opt_st, *args):
            diff_params, static_params = \
                eqx.partition(params, filter_spec)
            loss, grads = \
                self.loss_and_grads(diff_params, static_params, *args)
            updates, opt_st = self.opt.update(grads, opt_st)
            params = eqx.apply_updates(params, updates)
            return params, opt_st, loss
        return step
