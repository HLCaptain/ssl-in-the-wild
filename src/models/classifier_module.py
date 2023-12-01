from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torchvision
from torch import nn
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from src.models.vicreg_module import VICRegModule

class ClassifierModule(LightningModule):
    """Classification `LightningModule` for the BIRDS 525 SPECIES- IMAGE CLASSIFICATION dataset.
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        backbone_ckpt_path: str,
        num_classes : int = 525,
        max_epochs: int = 100
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        ckpt_path = backbone_ckpt_path
        model = VICRegModule.load_from_checkpoint(ckpt_path)
        model.eval()

        # use the pretrained ResNet backbone
        self.backbone = model.backbone
        self.max_epochs = max_epochs

        # freeze the backbone
        deactivate_requires_grad(self.backbone)

        # create a linear layer for our downstream classification model
        self.net = net
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        y_hat = self.backbone(x)
        y_hat = self.net(y_hat)
        return y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int)-> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y, _ = batch
        # x is list[torch.Tensor]
        y_hat = self.forward(x[0])
        loss = self.criterion(y_hat, y)
        self.log("classifier_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y, _ = batch
        y_hat = self.forward(x[0])
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # calculate and log top1 accuracy
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y, _ = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        return accuracy

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # calculate and log top1 accuracy
        if self.test_step_outputs:
            acc = self.test_step_outputs.sum() / len(self.test_step_outputs)
            self.log("test_acc", acc, on_epoch=True, prog_bar=True)
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        # optim = torch.optim.SGD(self.net.parameters(), lr=30.0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        # return [self.optimizer], [self.scheduler]
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "classifier_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ClassifierModule(None, None, None)
