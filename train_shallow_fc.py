import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, roc_auc_score)
from torch.utils.data import DataLoader

from h5dataset import H5Dataset
from shallow_fc import FCN

logger = logging.getLogger(__name__)

MAX_EPOCHS = 20
P_DROPOUT = [0, 0.2, 0.2]
LR = 1e-3
BATCH_SIZE = 64
NOISE_SCALE = 0.05
N_JOBS = 12


def transform(x: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian noise augmentation

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Output tensor
    """
    x += NOISE_SCALE * torch.randn_like(x)
    return x


def compute_metrics(ds: H5Dataset, scores: List[torch.Tensor]) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute classification metrics after training

    Parameters
    ----------
    ds : H5Dataset
        Dataset containing labels a .y property
    scores : List[Any]
        Batches with predicted scores

    Returns
    -------
    pd.Series
        Metrics as keys and values as values
    """
    y_true = ds.y.detach().cpu().numpy()
    y_score = np.concatenate([
        batch.detach().cpu().numpy()
        for batch in scores
    ])
    assert len(y_score) == len(y_true)
    y_pred = (y_score >= 0.5).astype(int)

    mn_score_pos = y_score[y_true == 1].mean()
    sd_score_pos = y_score[y_true == 1].std()
    mn_score_neg = y_score[y_true == 0].mean()
    sd_score_neg = y_score[y_true == 0].std()

    results = pd.DataFrame(dict(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
    ))

    return pd.Series(dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        auroc=roc_auc_score(y_true, y_score),
        ap=average_precision_score(y_true, y_score),
        mean_score_pos=mn_score_pos,
        std_score_pos=sd_score_pos,
        mean_score_neg=mn_score_neg,
        std_score_neg=sd_score_neg,
    )), results


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('checkpoint_path', type=click.Path(file_okay=False))
@click.option('-d', '--n_dims', type=click.INT, default=1280,
              help='Number of input features.')
def main(
    train_path: Union[str, Path],
    valid_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    n_dims: int
):
    """Run training of shallow FC network on ghostnet output

    Parameters
    ----------
    train_path : Union[str, Path]
        Path to training set
    valid_path : Union[str, Path]
        Path to validation set
    checkpoint_path : Union[str, Path]
        Path where final checkpoint will be saved
    """
    logger.info('Start')
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        logger=None,
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=False
    )

    channels = [n_dims, 2*n_dims, 2*n_dims, 1]

    net = FCN(channels, P_DROPOUT, LR)

    logger.info('Loading data')
    train_ds = H5Dataset(train_path, load_num=16000,
                         transform=transform, mean=None)
    mn = train_ds.mean
    sd = train_ds.std
    valid_ds = H5Dataset(valid_path, load_num=1000, mean=mn, std=sd)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=N_JOBS
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=N_JOBS
    )

    logger.info('Training')
    trainer.fit(net, train_dl, valid_dl)

    logger.info('Saving model')
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    trainer.save_checkpoint(checkpoint_path / 'shallow_fc.pt')

    logger.info('Predict on validation dataset')
    scores = trainer.predict(net, valid_dl)

    metrics, predictions = compute_metrics(valid_ds, scores)
    logger.info(f'Metrics:\n{metrics}')

    metrics.to_csv(checkpoint_path / 'metrics.csv')
    predictions.to_csv(checkpoint_path / 'predictions.csv')

    logger.info('Done')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
