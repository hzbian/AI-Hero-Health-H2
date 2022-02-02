from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics.functional as tmf
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam


class FCN(pl.LightningModule):
    def __init__(
        self,
        channels: List[int],
        p_dropout: List[float],
        lr: float,
        weight_decay: float = 1e-4
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.metrics = dict(
            accuracy=tmf.accuracy,
            auroc=tmf.auroc,
            ap=tmf.average_precision,
        )

        layers: List[nn.Module] = []
        for in_ch, out_ch, p in zip(channels[:-1], channels[1:], p_dropout):
            layers.append(nn.Dropout(p))
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.ReLU())

        layers[-1] = nn.Sigmoid()
        layers.append(nn.Flatten(0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = binary_cross_entropy(y_hat, y.float())

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = binary_cross_entropy(y_hat, y.float())

        self.log('val_loss', loss, prog_bar=True)

        self.log_dict({
            name: metric(y_hat, y)
            for name, metric in self.metrics.items()
        }, prog_bar=True)

        return loss

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )


if __name__ == "__main__":
    net = FCN([10, 100, 1], [0, 0.2], 1e-3, 1e-4)
    trainer = pl.Trainer(accelerator='auto',
                         deterministic=True, max_epochs=10, logger=None,
                         enable_checkpointing=False, gpus=1)

    x = torch.randn(100, 10)
    y = torch.randint(low=0, high=2, size=(100,))
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=10)

    trainer.fit(net, train_dataloaders=dl, val_dataloaders=dl)
