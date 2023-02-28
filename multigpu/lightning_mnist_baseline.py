import os


import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class LitMNIST(LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            "test_acc",
            self.test_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED
    ####################


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = PATH_DATASETS, batch_size: int = 64, num_workers: int = 8
    ):
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
        self.setup()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
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


def main():
    seed_everything(1)

    data_module = MNISTDataModule()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    model = LitMNIST(learning_rate=1e-3)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc_epoch",
        save_top_k=3,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(
        project="lightning_MNIST_test",  # group runs in "MNIST" project
        # log_model="all",
        save_dir="./experiments/logs",
        tags=["MNIST", "baseline"],
    )

    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            lr_monitor,
        ],
        deterministic=True,
        strategy="ddp"
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
