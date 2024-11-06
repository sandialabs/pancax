from abc import abstractmethod
import equinox as eqx


class BaseLossFunction(eqx.Module):
  """
  Base class for loss functions. 
  Currently does nothing but helps build a 
  type hierarchy.
  """
  
  def filtered_loss(self, diff_params, static_params, domain):
    params = eqx.combine(diff_params, static_params)
    return self.__call__(params, domain) 


class BCLossFunction(BaseLossFunction):
  """
  Base class for boundary condition loss functions.

  A ``load_step`` method is expect with the following
  type signature
  ``load_step(self, params, domain, t)``
  """
  @abstractmethod
  def load_step(self, params, domain, t):
    pass


class PhysicsLossFunction(BaseLossFunction):
  """
  Base class for physics loss functions.

  A ``load_step`` method is expect with the following
  type signature
  ``load_step(self, params, domain, t)``
  """
  @abstractmethod
  def load_step(self, params, domain, t):
    pass
