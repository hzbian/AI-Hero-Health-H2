import logging
from pathlib import Path
from typing import Union

import click
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from ..dataset import CovidImageDataset

logger = logging.getLogger(__name__)


def ds_to_numpy(ds: Dataset) -> np.ndarray:

    return np.concatenate([
        img.detach().cpu().numpy().flatten()[np.newaxis, ]
        for img in ds
    ])


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('-n', '--n_dimensions', type=click.INT, default=500,
              help='Number of kept PCA dimensions')
@click.option('-s', '--seed', type=click.INT, default=137,
              help='PRNG seed')
def main(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_dimensions: int,
    seed: int
):
    data_dir = Path(data_dir)

    root_dir = data_dir / 'imgs'
    train_fn = data_dir / 'train.csv'
    valid_fn = data_dir / 'valid.csv'

    train_ds = CovidImageDataset(train_fn, root_dir)
    valid_ds = CovidImageDataset(valid_fn, root_dir)
    logger.info(f'{len(train_ds)=}')
    logger.info(f'{len(valid_ds)=}')

    train = ds_to_numpy(train_ds)
    valid = ds_to_numpy(valid_ds)
    logger.info(f'{train.shape=}')
    logger.info(f'{valid.shape=}')

    pca = PCA(n_components=n_dimensions, random_state=seed)
    train_pca = pca.fit_transform(train)
    valid_pca = pca.transform(valid)
    logger.info(f'{train_pca.shape=}')
    logger.info(f'{valid_pca.shape=}')

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_fn = output_dir / 'train.npy'
    valid_fn = output_dir / 'valid.npy'
    np.save(train_fn, train_pca)
    np.save(valid_fn, valid_pca)

    pca_fn = output_dir / 'pca.joblib'
    dump(pca, pca_fn)


if __name__ == "__main__":
    main()
