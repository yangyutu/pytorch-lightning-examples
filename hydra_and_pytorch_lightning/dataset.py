import logging
import glob
import os
import torch
import numpy as np
import gzip
import random
import pytorch_lightning as pl

from typing import Tuple, List, Optional
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 8):
        """_summary_

        Args:
            text_data_name (str): the name of the dataset from text claasification dataset in https://pytorch.org/text/stable/datasets.html
            vocab_min_freq (int, optional): minimum token frequency to count the token as a valid entry in the dictionary. Defaults to 1.
            batch_size (int, optional): batch size for dataloader. Defaults to 64.
            num_workers (int, optional): number of worker for dataloader. Defaults to 8.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
