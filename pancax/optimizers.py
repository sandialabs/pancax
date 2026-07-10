from typing import Any, Callable, Optional
import equinox as eqx
import jax.numpy as jnp
import optax
import optimistix as optx


class OptimizerState(eqx.Module):
    opt_state: Any
    epoch: Any
    filter_spec: Any
    is_ensemble: bool


class AbstractOptimizer(eqx.Module):
    loss_function: Callable
    # opt: optax.GradientTransformation = eqx.field(static=True)
    opt: any
    has_aux: bool
    jit: bool
    ensemble_step: any
    single_step: any

    def __init__(
        self,
        loss_function: Any,
        opt: optax.GradientTransformation,
        *,
        has_aux: bool = False,
        jit: bool = True,
    ):
        self.loss_function = loss_function
        self.opt = opt
        self.has_aux = has_aux
        self.jit = jit

        if self.jit:
            self.ensemble_step = eqx.filter_jit(self._ensemble_step)
            self.single_step = eqx.filter_jit(self._single_step)
        else:
            self.ensemble_step = self._ensemble_step
            self.single_step = self._single_step

    def init(
        self,
        params,
        filter_spec: Optional[Any] = None
    ):
        if filter_spec is None:
            filter_spec = params.freeze_physics_normalization_filter()

        is_ensemble = params.is_ensemble

        trainable_params = self._trainable_params(params, filter_spec)
        opt_state = self.opt.init(trainable_params)

        return OptimizerState(
            opt_state=opt_state,
            epoch=jnp.asarray(0),
            filter_spec=filter_spec,
            is_ensemble=is_ensemble,
        )

    def step(self, params, state: OptimizerState, *args):
        if state.is_ensemble:
            return self.ensemble_step(params, state, *args)
        else:
            return self.single_step(params, state, *args)

    @staticmethod
    def _trainable_params(params, filter_spec):
        return eqx.filter(
            eqx.filter(params, filter_spec),
            eqx.is_inexact_array,
        )

    def _single_step(self, params, state: OptimizerState, *args):
        filter_spec = state.filter_spec
        diff_params, static_params = eqx.partition(params, filter_spec)
        loss_and_grads = eqx.filter_value_and_grad(
            self.loss_function.filtered_loss,
            has_aux=True,
        )
        loss, grads = loss_and_grads(
            diff_params,
            static_params,
            *args,
        )
        grads = self._trainable_params(grads, filter_spec)
        trainable_params = self._trainable_params(params, filter_spec)

        updates, opt_state = self.opt.update(
            grads,
            state.opt_state,
            trainable_params,
        )

        params = eqx.apply_updates(params, updates)

        state = OptimizerState(
            opt_state=opt_state,
            epoch=state.epoch + 1,
            filter_spec=state.filter_spec,
            is_ensemble=state.is_ensemble,
        )

        return params, state, loss

    def _ensemble_step(self, params, state: OptimizerState, *args):
        filter_spec = state.filter_spec
        diff_params, static_params = eqx.partition(params, filter_spec)
        loss_and_grads = eqx.filter_vmap(
            eqx.filter_value_and_grad(
                self.loss_function.filtered_loss,
                has_aux=self.has_aux
            ),
            in_axes=(eqx.if_array(0), eqx.if_array(0), None)
        )
        loss, grads = loss_and_grads(diff_params, static_params, *args)

        grads = self._trainable_params(grads, filter_spec)
        trainable_params = self._trainable_params(params, filter_spec)

        updates, opt_state = self.opt.update(
            grads,
            state.opt_state,
            trainable_params,
        )

        params = eqx.apply_updates(params, updates)

        state = OptimizerState(
            opt_state=opt_state,
            epoch=state.epoch + 1,
            filter_spec=state.filter_spec,
            is_ensemble=state.is_ensemble,
        )

        return params, state, loss


class Adam(AbstractOptimizer):
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
        # TODO figure out how best to handle scheduler
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )

        if clip_gradients:
            opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )
        else:
            opt = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )

        super().__init__(
            loss_function,
            opt,
            has_aux=has_aux,
            jit=jit
        )


class LBFGSOptState(eqx.Module):
    solver_state: Any
    done: Any
    result: Any


class LBFGS(AbstractOptimizer):
    options: Any
    tags: Any

    def __init__(
        self,
        loss_function: Callable,
        has_aux: Optional[bool] = False,
        jit: Optional[bool] = True,
        rtol: Optional[float] = 1.0e-3,
        atol: Optional[float] = 1.0e-3,
        options: Optional[dict[str, Any]] = None,
        tags: Optional[frozenset] = None,
    ) -> None:
        opt = optx.LBFGS(
            rtol=rtol,
            atol=atol,
        )
        self.options = {} if options is None else options
        self.tags = frozenset() if tags is None else tags
        super().__init__(
            loss_function,
            opt,
            has_aux=has_aux,
            jit=jit,
        )

    def init(
        self,
        params,
        filter_spec: Optional[Any] = None
    ):
        if filter_spec is None:
            filter_spec = params.freeze_physics_normalization_filter()

        is_ensemble = params.is_ensemble

        opt_state = LBFGSOptState(
            solver_state=None,
            done=jnp.asarray(False),
            result=None,
        )

        return OptimizerState(
            opt_state=opt_state,
            epoch=jnp.asarray(0),
            filter_spec=filter_spec,
            is_ensemble=is_ensemble,
        )

    def _init_solver_state(
        self,
        trainable_params,
        params,
        filter_spec,
        *args,
    ):
        objective_args = (params, filter_spec, args)
        f_struct, aux_struct = eqx.filter_eval_shape(
            self._objective,
            trainable_params,
            objective_args,
        )
        solver_state = self.opt.init(
            self._objective,
            trainable_params,
            objective_args,
            self.options,
            f_struct,
            aux_struct,
            self.tags,
        )
        done, result = self.opt.terminate(
            self._objective,
            trainable_params,
            objective_args,
            self.options,
            solver_state,
            self.tags,
        )

        return LBFGSOptState(
            solver_state=solver_state,
            done=done,
            result=result,
        )

    def _objective(self, trainable_params, args):
        params, filter_spec, args = args
        params = eqx.combine(trainable_params, params)
        diff_params, static_params = eqx.partition(params, filter_spec)
        return self.loss_function.filtered_loss(
            diff_params, static_params, *args
        )

    def _single_step_old(self, params, state: OptimizerState, *args):
        filter_spec = state.filter_spec

        trainable_params = self._trainable_params(
            params,
            filter_spec,
        )

        lbfgs_state = state.opt_state

        if lbfgs_state.solver_state is None:
            lbfgs_state = self._init_solver_state(
                trainable_params,
                params,
                filter_spec,
                *args,
            )

        def objective(trainable_params_, objective_args):
            params_, filter_spec_, loss_args_ = objective_args

            return self._loss_from_trainable_params(
                trainable_params_,
                params_,
                filter_spec_,
                *loss_args_,
            )

        objective_args = (
            params,
            filter_spec,
            args,
        )

        new_trainable_params, new_solver_state, _aux = self.opt.step(
            objective,
            trainable_params,
            objective_args,
            self.options,
            lbfgs_state.solver_state,
            self.tags,
        )

        new_params = eqx.combine(
            new_trainable_params,
            params,
        )

        new_objective_args = (
            new_params,
            filter_spec,
            args,
        )

        done, result = self.opt.terminate(
            objective,
            new_trainable_params,
            new_objective_args,
            self.options,
            new_solver_state,
            self.tags,
        )

        new_lbfgs_state = LBFGSOptState(
            solver_state=new_solver_state,
            done=done,
            result=result,
        )

        new_state = OptimizerState(
            opt_state=new_lbfgs_state,
            epoch=state.epoch + 1,
            filter_spec=state.filter_spec,
            is_ensemble=state.is_ensemble,
        )

        loss = self._report_loss(
            new_params,
            filter_spec,
            *args,
        )

        return new_params, new_state, loss

    def _single_step(self, params, state: OptimizerState, *args):
        filter_spec = state.filter_spec
        lbfgs_state = state.opt_state
        trainable_params = self._trainable_params(params, filter_spec)
        if lbfgs_state.solver_state is None:
            lbfgs_state = self._init_solver_state(
                trainable_params, params,
                filter_spec, *args
            )

        objective_args = (params, filter_spec, args)
        trainable_params, solver_state, aux = self.opt.step(
            self._objective,
            trainable_params,
            objective_args,
            self.options,
            lbfgs_state.solver_state,
            self.tags
        )
        done, result = self.opt.terminate(
            self._objective,
            trainable_params,
            objective_args,
            self.options,
            solver_state,
            self.tags
        )
        lbfgs_state = LBFGSOptState(
            solver_state=solver_state,
            done=done,
            result=result,
        )
        state = OptimizerState(
            opt_state=lbfgs_state,
            epoch=state.epoch + 1,
            filter_spec=state.filter_spec,
            is_ensemble=state.is_ensemble,
        )
        params = eqx.combine(trainable_params, params)
        objective_args = (params, filter_spec, args)
        loss = self._objective(trainable_params, objective_args)
        return params, state, loss

    def _ensemble_step(self, params, state: OptimizerState, *args):
        raise NotImplementedError(
            "Per-ensemble LBFGS is not implemented here. "
            "Optimistix LBFGS minimizes one scalar objective at a time, so "
            "independent ensemble members need independently vmapped solver "
            "states. The single-model path is implemented."
        )
