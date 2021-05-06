import logging
import os
import config as cfg
import torch
from torch.utils.data import Dataset, DataLoader
from utils.log import create_logger
from torch.optim.optimizer import Optimizer


class DataWrapper(Dataset):
    """
    A wrapper for torch.utils.data.Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Instructor:
    """
    A base class for training.
    """
    def __init__(self, cmd):
        """
        Args:
            cmd: command line parameters.
        """
        self.log = create_logger(
            __name__, silent=False,
            to_disk=True, log_file=cfg.log)
        self.cmd = cmd
        self.log.info(str(cmd))

    def rename_log(self, filename: str):
        """
        Renaming the log file.
        """
        logging.shutdown()
        os.rename(cfg.log, filename)

    @staticmethod
    def optimize(opt: Optimizer, loss: torch.Tensor):
        """
        Optimize the parameters based on the loss and the optimizer.

        Args:
            opt: optimizer
            loss: loss, a scalar
        """
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(inputs, batch_size: int, shuffle: bool=True):
        """
        Return a dataloader given the input and the batch size.
        """
        data = DataWrapper(inputs)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle)
        return batches
