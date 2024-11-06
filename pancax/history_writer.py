from abc import abstractmethod
from pathlib import Path
from typing import List
import equinox as eqx
import pandas


class BaseHistoryWriter(eqx.Module):
  log_every: int
  write_every: int

  @abstractmethod
  def to_csv(self):
    pass

  def write_data(self, key: str, val: float) -> None:
    if key in self.data_dict.keys():
      self.data_dict[key].append(val)
    else:
      self.data_dict[key] = [val]

  @abstractmethod
  def write_aux_values(self, loss):
    pass

  @abstractmethod
  def write_epoch(self, epoch):
    pass

  def write_history(self, loss, epoch: int):
    if epoch % self.log_every == 0:
      self.write_aux_values(loss)
      self.write_epoch(epoch)
      self.write_loss(loss)

    if epoch % self.write_every == 0:
      self.to_csv()

  @abstractmethod
  def write_loss(self, loss):
    pass


class HistoryWriter(BaseHistoryWriter):
  log_every: int
  write_every: int
  data_dict: dict
  history_file: Path

  def __init__(self, history_file: str, log_every: int, write_every: int) -> None:
    super().__init__(log_every, write_every)
    self.history_file = Path(history_file)
    self.data_dict = dict()

  def to_csv(self) -> None:
    df = pandas.DataFrame(self.data_dict)
    df.to_csv(self.history_file, index=False)

  def write_aux_values(self, loss):
    for key, val in loss[1].items():
      if key == 'props':
        for prop_num, prop_val in enumerate(val):
          self.write_data(f'property_{prop_num}', prop_val.item())
      elif key == 'dprops':
        for prop_num, prop_val in enumerate(val):
          self.write_data(f'dproperty_{prop_num}', prop_val.item())
      else:
        self.write_data(key, val.item())

  def write_epoch(self, epoch):
    self.write_data('epoch', epoch)

  def write_loss(self, loss):
    self.write_data('loss', loss[0].item())


  def _write_loss(self, loss, epoch: int, log_every=1, save_every=1000):
    if epoch % log_every == 0:
      self.write_data('epoch', epoch)
      self.write_data('loss', loss[0].item())
      for key, val in loss[1].items():
        if key == 'props':
          for prop_num, prop_val in enumerate(val):
            self.write_data(f'property_{prop_num}', prop_val.item())
        elif key == 'dprops':
          for prop_num, prop_val in enumerate(val):
            self.write_data(f'dproperty_{prop_num}', prop_val.item())
        else:
          self.write_data(key, val.item())

    if epoch % save_every == 0:
      print(f'Saving history buffer to {self.history_file}')
      self.to_csv(self.history_file)


class EnsembleHistoryWriter(BaseHistoryWriter):
  log_every: int
  write_every: int
  history_writers: List[HistoryWriter]

  def __init__(self, base_name: str, n_pinns: int, log_every: int, write_every: int) -> None:
    super().__init__(log_every, write_every)
    self.history_writers = [
      HistoryWriter(f'{base_name}_{n}.csv', log_every, write_every) for n in range(n_pinns)
    ]
    
  def to_csv(self):
    for writer in self.history_writers:
      writer.to_csv()

  def write_aux_values(self, loss):
    for key, val in loss[1].items():
      for n, writer in enumerate(self.history_writers):
        if key == 'props':
          # for prop_num, prop_val in enumerate(val):
          #   writer.write_data(f'property_{prop_num}', prop_val[n].item())
          # writer.write_data(f'property_{prop_num}')
          for prop_num, prop_val in enumerate(val[n]):
            writer.write_data(f'property_{prop_num}', prop_val.item())
          # writer.write_data(f'property_')
        else:
          writer.write_data(key, val[n].item())

  def write_epoch(self, epoch):
    for writer in self.history_writers:
      writer.write_epoch(epoch)

  def write_loss(self, loss):
    for n, writer in enumerate(self.history_writers):
      writer.write_data('loss', loss[0][n].item())
