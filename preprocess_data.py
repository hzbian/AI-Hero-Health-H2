import logging
from pathlib import Path
from typing import Tuple, Union

import click
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from umap import UMAP

from dataset import CovidImageDataset

logger = logging.getLogger(__name__)


def ds_to_numpy(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Transform torch dataset to numpy arrays

    Parameters
    ----------
    ds : Dataset
        COVID dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Flattened images + labels
    """

    imgs = []
    labels = []
    for img, label in ds:
        imgs.append(img.detach().cpu().numpy().flatten()[np.newaxis, ])
        labels.append(label)

    return np.concatenate(imgs), np.array(labels)


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('-n', '--n_dimensions', type=click.INT, default=500,
              help='Number of kept PCA dimensions')
@click.option('-s', '--seed', type=click.INT, default=137,
              help='PRNG seed')
@click.option('-p/-np', '--pca/--no-pca', default=True, is_flag=True)
@click.option('-u/-nu', '--umap/--no-umap', default=False, is_flag=True,)
def main(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_dimensions: int,
    seed: int,
    pca: bool,
    umap: bool,
):
    """Project image using PCA

    Load raw data from DATA_DIR, perform PCA 
    and save results in OUTPUT_DIR

    \f
    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing raw data
    output_dir : Union[str, Path]
        Output directory
    n_dimensions : int
        Dimensions kept of the PCA
    seed : int
        PRNG seed.
    """
    data_dir = Path(data_dir)

    root_dir = data_dir / 'imgs'
    train_fn = data_dir / 'train.csv'
    valid_fn = data_dir / 'valid.csv'

    train_ds = CovidImageDataset(train_fn, root_dir)
    valid_ds = CovidImageDataset(valid_fn, root_dir)
    logger.info(f'{len(train_ds)=}')
    logger.info(f'{len(valid_ds)=}')

    train, train_labels = ds_to_numpy(train_ds)
    valid, valid_labels = ds_to_numpy(valid_ds)
    logger.info(f'{train.shape=}')
    logger.info(f'{valid.shape=}')

    train_df: pd.DataFrame = train_ds.info_df
    valid_df: pd.DataFrame = valid_ds.info_df

    train_df['label'] = train_labels
    valid_df['label'] = valid_labels

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_fn = output_dir / 'train.csv'
    valid_fn = output_dir / 'valid.csv'
    train_df.to_csv(train_fn, index=False)
    valid_df.to_csv(valid_fn, index=False)

    if pca:
        logger.info('Applying PCA')
        pca = PCA(n_components=n_dimensions, random_state=seed)
        train_pca = pca.fit_transform(train)
        valid_pca = pca.transform(valid)
        logger.info(f'{train_pca.shape=}')
        logger.info(f'{valid_pca.shape=}')

        train_fn = output_dir / 'train.npy'
        valid_fn = output_dir / 'valid.npy'
        np.save(train_fn, train_pca)
        np.save(valid_fn, valid_pca)

        pca_fn = output_dir / 'pca.joblib'
        dump(pca, pca_fn)

    if umap:
        logger.info('Applying UMAP')
        umap = UMAP(n_components=min(n_dimensions, 100))
        train_umap = umap.fit_transform(train)
        valid_umap = umap.transform(valid)
        logger.info(f'{train_umap.shape=}')
        logger.info(f'{valid_umap.shape=}')

        np.save(output_dir / 'train_umap.npy', train_umap)
        np.save(output_dir / 'valid_umap.npy', valid_umap)
        dump(umap, output_dir / 'umap.joblib')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
