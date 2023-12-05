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
        backbone_ckpt_path: str,
        ssl_backbone: bool = True,
        backbone_pretrained: bool = True,
        freeze: bool = True,
        num_classes : int = 525,
        max_epochs: int = 100
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if(ssl_backbone):
            # use the pretrained VICReg backbone
            ckpt_path = backbone_ckpt_path
            model = VICRegModule.load_from_checkpoint(ckpt_path)
            model.eval()
            self.backbone = model.backbone
        else:
            # without SSL pretraining
            resnet = torchvision.models.resnet18(pretrained=backbone_pretrained)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone = backbone
        
        self.max_epochs = max_epochs

        if(freeze):
            # freeze the backbone
            deactivate_requires_grad(self.backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
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
        y_hat = self.forward(x)
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
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


if __name__ == "__main__":
    _ = ClassifierModule(None, None, None)