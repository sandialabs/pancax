from .base_loss_function import BaseLossFunction


class CombineLossFunctions(BaseLossFunction):
  funcs: any
  with_props: bool

  def __init__(self, *funcs, with_props=False):
    temp_funcs = []
    for f in funcs:
      temp_funcs.append(f)
    self.funcs = funcs
    self.with_props = with_props

  def __call__(self, params, domain):
    loss = 0.0
    aux = dict()
    for f in self.funcs:
      temp_loss, temp_aux = f(params, domain)
      loss = loss + temp_loss
      aux.update(temp_aux)

    if self.with_props:
      props = params.properties()
      aux.update({'props': props})

    return loss, aux
