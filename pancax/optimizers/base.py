from abc import ABC
from abc import abstractmethod
from pancax.logging import log_loss
from typing import Callable
from typing import Optional
import equinox as eqx


class Optimizer(ABC):
  def __init__(
    self,
    loss_function: Callable,
    has_aux: Optional[bool] = False,
    jit: Optional[bool] = True
  ) -> None:
    self.loss_function = loss_function
    self.has_aux = has_aux
    self.jit = jit
    self.step = None
    self.epoch = 0

  @abstractmethod
  def make_step_method(self, params):
    pass

  def ensemble_init(self, params):
    self.step = self.make_step_method()
    # if self.jit:
    #   self.step = eqx.filter_jit(self.step)

    # need to now make an ensemble wrapper our self.step
    # but make sure not to jit it until after the vmap
    def ensemble_step(params, domain, opt_st):
      params, opt_st, loss = eqx.filter_vmap(
        self.step, in_axes=(eqx.if_array(0), None, eqx.if_array(0))
      )(params, domain, opt_st)
      return params, opt_st, loss
    
    if self.jit:
      self.ensemble_step = eqx.filter_jit(ensemble_step)

    def vmap_func(p):
      return self.opt.init(eqx.filter(p, eqx.is_array))
    
    opt_st = eqx.filter_vmap(vmap_func, in_axes=(eqx.if_array(0),))(params)
    return opt_st

  def ensemble_step_old(self, params, domain, opt_st):
    params, opt_st, loss = eqx.filter_vmap(
      self.step, in_axes=(eqx.if_array(0), None, eqx.if_array(0))
    )(params, domain, opt_st)
    return params, opt_st, loss

  def init(self, params):
    self.step = self.make_step_method()
    if self.jit:
      self.step = eqx.filter_jit(self.step)

    opt_st = self.opt.init(eqx.filter(params, eqx.is_array))
    # opt_st = self.opt.init(eqx.filter(params, filter_spec))
    return opt_st

  # def train(self, params, opt_st, domain, n_epochs, log_every):
  #   for n in range(n_epochs):
  #     params, opt_st, loss = self.step(params, domain, opt_st)
  #     log_loss(loss, self.epoch, log_every)
  #     self.epoch = self.epoch + 1
  #   return params, opt_st
  def train(
    self, 
    params, domain, times, opt, logger, history, pp,
    n_epochs,
    log_every: Optional[int] = 100,
    serialise_every: Optional[int] = 10000,
    postprocess_every: Optional[int] = 10000
  ):
    opt_st = opt.init(params)
    for epoch in range(int(n_epochs)):
      params, opt_st, loss = opt.step(params, domain, opt_st)
      logger.log_loss(loss, epoch, log_every)
      history.write_loss(loss, epoch)

      if epoch % serialise_every == 0:
        params.serialise('checkpoint', epoch)

      if epoch % postprocess_every == 0:
        pp.init(params, domain, f'output_{str(epoch).zfill(6)}.e')
        pp.write_outputs(params, domain, times, [
          'displacement',
        ])
        pp.close()
