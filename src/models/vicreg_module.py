from typing import Any, Dict, Tuple

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torchvision
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)

class VICRegModule(LightningModule):
    """`LightningModule` for the BIRDS 525 SPECIES- IMAGE CLASSIFICATION dataset.
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
    ) -> None:
        """Initialize a `VICRegModule`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=2048,
            output_dim=2048,
            num_layers=4,
        )

        # loss function
        self.criterion = VICRegLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int)-> torch.Tensor:
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("vicreg_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

if __name__ == "__main__":
    _ = VICRegModule()
