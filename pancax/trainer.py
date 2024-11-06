from pancax.history_writer import HistoryWriter
from pancax.logging import Logger
from pancax.post_processor import PostProcessor
from pancax.utils import set_checkpoint_file
from pathlib import Path
import os

# TODO make this a proper equinox module
class Trainer:
    def __init__(
        self,
        domain,
        opt,
        checkpoint_base = 'checkpoint',
        history_file = 'history.csv',
        log_file = 'pinn.log',
        output_file_base = 'output',
        output_node_variables = [],
        output_element_variables = [],
        log_every = 1000,
        output_every = 10000,
        serialise_every = 10000,
        workdir = os.getcwd()
    ) -> None:
        try:
            os.makedirs(Path(os.path.join(workdir, 'checkpoint')))
            os.makedirs(Path(os.path.join(workdir, 'history')))
            os.makedirs(Path(os.path.join(workdir, 'log')))
            os.makedirs(Path(os.path.join(workdir, 'results')))
        except FileExistsError:
            pass

        self.domain = domain
        self.opt = opt
        self.checkpoint_base = Path(os.path.join(workdir, 'checkpoint', checkpoint_base))
        self.serialise_every = serialise_every
        self.history = HistoryWriter(Path(os.path.join(workdir, 'history', history_file)), log_every, log_every)
        self.logger = Logger(Path(os.path.join(workdir, 'log', log_file)), log_every)
        self.output_every = output_every
        self.output_file_base = Path(os.path.join(workdir, 'results', output_file_base))
        self.output_node_variables = output_node_variables
        self.output_element_variables = output_element_variables
        self.pp = PostProcessor(domain.mesh_file)
        self.epoch = 0

    def init(self, params):
        opt_st = self.opt.init(params)
        return opt_st

    def serialise(self, params):
        if self.epoch % self.serialise_every == 0:
            params.serialise(self.checkpoint_base, self.epoch)

    def step(self, params, opt_st):
        params, opt_st, loss = self.opt.step(params, self.domain, opt_st)
        self.history.write_history(loss, self.epoch)
        self.logger.log_loss(loss, self.epoch)
        self.serialise(params)
        self.write_outputs(params)
        self.epoch = self.epoch + 1
        return params, opt_st

    def train(self, params, n_epochs):
        opt_st = self.init(params)
        for epoch in range(n_epochs):
            params, opt_st = self.step(params, opt_st)
        return params

    def write_outputs(self, params):
        if self.epoch % self.output_every == 0:
            self.pp.init(
                self.domain, 
                f'{self.output_file_base}_{str(self.epoch).zfill(8)}.e',
                node_variables=self.output_node_variables,
                element_variables=self.output_element_variables
            )
            self.pp.write_outputs(params, self.domain)
            self.pp.close()
