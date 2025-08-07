from abc import abstractmethod
from typing import Callable, Optional, Union
import equinox as eqx


class AbstractOptimizer(eqx.Module):
    epoch: int
    has_aux: bool
    jit: bool
    loss_function: Callable  # TODO further type me
    loss_and_grads: Callable  # TODO not sure if this type is right?
    step: Union[None, Callable]

    def __init__(
        self,
        loss_function: Callable,
        *,
        has_aux: Optional[bool] = False,
        jit: Optional[bool] = True
    ) -> None:
        self.epoch = 0
        self.has_aux = has_aux
        self.jit = jit
        self.loss_function = loss_function
        self.loss_and_grads = eqx.filter_value_and_grad(
            self.loss_function.filtered_loss, has_aux=has_aux
        )
        self.step = None

    @abstractmethod
    def make_step_method(self, filter_spec: Callable):
        pass

    def _ensemble_init(self, params, filter_spec):
        filter_spec = self._init_filter(params, filter_spec=filter_spec)
        step = self.make_step_method(filter_spec)

        def ensemble_step(params, opt_st, *args):
            in_axes = (eqx.if_array(0), eqx.if_array(0))
            in_axes = in_axes + len(args) * (None,)

            @eqx.filter_vmap(in_axes=in_axes)
            def vmap_func(params, opt_st, *args):
                return step(params, opt_st, *args)

            return vmap_func(params, opt_st, *args)

        if self.jit:
            ensemble_step = eqx.filter_jit(ensemble_step)

        self = eqx.tree_at(
            lambda x: x.step, self, ensemble_step,
            is_leaf=lambda x: x is None
        )

        @eqx.filter_vmap(in_axes=(eqx.if_array(0),))
        def vmap_func(p):
            return self.opt.init(eqx.filter(p, eqx.is_array))

        opt_st = vmap_func(params)

        return self, opt_st

    def _init(self, params, filter_spec):
        filter_spec = self._init_filter(params, filter_spec=filter_spec)
        step = self.make_step_method(filter_spec)
        if self.jit:
            step = eqx.filter_jit(step)

        self = eqx.tree_at(
            lambda x: x.step, self, step,
            is_leaf=lambda x: x is None
        )
        opt_st = self.opt.init(eqx.filter(params, eqx.is_array))
        return self, opt_st

    def init(self, params, filter_spec: Optional[Callable] = None):
        if params.is_ensemble:
            return self._ensemble_init(params, filter_spec)
        else:
            return self._init(params, filter_spec)

    def _init_filter(self, params, filter_spec):
        if filter_spec is None:
            filter_spec = params.freeze_physics_normalization_filter()

        return filter_spec
