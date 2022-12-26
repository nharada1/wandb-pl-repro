import glob
import os
import setuptools
import uuid

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
import segmentation_models_pytorch as smp

BATCH_SIZE = 32

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        aux_params=dict(
            pooling='avg',
            activation='softmax',
            classes=100,
        )
        self.model = smp.Unet(
            encoder_name="resnet50",
            in_channels=3,
            classes=10,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        _, y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        _, y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def on_validation_end(self):
        print(f"Final global step is {self.global_step}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def get_trainer(train, prefix):
    # Init our model
    mnist_model = Model()

    # Init DataLoader from MNIST Dataset
    train_ds = FakeData(size=10000, image_size=[3, 128, 128], num_classes=10, transform=transforms.ToTensor())
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    postfix = 'train' if train else 'val'
    name = f"{prefix}-{postfix}"
    wandb_logger = pl.loggers.WandbLogger(project='project-debug', name=name)

    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="/tmp/pl_wandb",
        save_top_k=-1,
        verbose=True,
    )

    callbacks = []
    if train:
        callbacks = [epoch_checkpoint_callback]

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=5,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    return mnist_model, loader, trainer

def main():
    prefix = str(uuid.uuid4()).split('-')[0]

    # Train the model
    model, data, trainer = get_trainer(train=True, prefix=prefix)
    trainer.fit(model, data)

    # Eval the model
    checkpoint_file = sorted(glob.glob(os.path.join("/tmp/pl_wandb/*.ckpt")))[-1]
    model, data, trainer = get_trainer(train=False, prefix=prefix)
    trainer.validate(model, data, ckpt_path=checkpoint_file)


if __name__ == "__main__":
    main()