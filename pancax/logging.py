from abc import abstractmethod
from pathlib import Path
from typing import List
import equinox as eqx


def log_loss(loss, n, log_every):
  if n % log_every == 0:
    print(f'Epoch {n}:')
    print(f'\tLoss = {loss[0].item()}')
    for key, val in loss[1].items():
      if key == 'props' or key == 'dprops':
        print(f'\t{key} = {val}')
      else:
        print(f'\t{key} = {val.item()}')


class BaseLogger(eqx.Module):
  log_every: int

  @abstractmethod
  def flush(self):
    pass

  def log_loss(self, loss, epoch):
    if epoch % self.log_every == 0:
      self.write_epoch_value(epoch)
      self.write_loss_value(loss)
      self.write_aux_values(loss)
      self.flush()

  @abstractmethod
  def write_aux_values(self, loss):
    pass

  @abstractmethod
  def write_epoch_value(self, epoch):
    pass

  @abstractmethod
  def write_loss_value(self, loss):
    pass


class Logger(BaseLogger):
  log_file: any

  def __init__(self, log_file_in: str, log_every: int) -> None:
    super().__init__(log_every)
    log_file_in = Path(log_file_in)
    self.log_file = open(log_file_in, 'w')

  def __exit__(self, exc_type, exc_value, exc_traceback):
    print(f'Closing log file.')
    self.log_file.close()

  def flush(self):
    self.log_file.flush()

  def write_aux_values(self, loss):
    for key, val in loss[1].items():
      if key == 'props' or key == 'dprops':
        self.log_file.write(f'  {key} = {val}\n')
      else:
        self.log_file.write(f'  {key} = {val.item()}\n')

  def write_epoch_value(self, epoch):
    self.log_file.write(f'Epoch {epoch}:\n')

  def write_loss_value(self, loss):
    self.log_file.write(f'  Loss = {loss[0].item()}\n')


class EnsembleLogger(BaseLogger):
  loggers: List[Logger]

  def __init__(self, base_name: str, n_pinns: int, log_every: int) -> None:
    super().__init__(log_every)
    self.loggers = [Logger(f'{base_name}_{n}.log', log_every) for n in range(n_pinns)]

  def flush(self):
    for logger in self.loggers:
      logger.flush()

  def write_aux_values(self, loss):
    for key, val in loss[1].items():
      for n, logger in enumerate(self.loggers):
        if key == 'props' or key == 'dprops':
          logger.log_file.write(f'  {key} = {val[n]}\n')
        else:
          logger.log_file.write(f'  {key} = {val[n].item()}\n')

  def write_epoch_value(self, epoch):
    for logger in self.loggers:
      logger.write_epoch_value(epoch)

  def write_loss_value(self, loss):
    for n, val in enumerate(loss[0]):
      self.loggers[n].log_file.write(f'  Loss = {val.item()}\n')
