import logging
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, roc_auc_score)

logger = logging.getLogger(__name__)

MODELS = {
    'logistic_regression': (LogisticRegression, dict(C=1.0, n_jobs=20)),
    'random_forest': (RandomForestClassifier,
                      dict(n_estimators=100, n_jobs=20)),
}


def get_model(model: str) -> ClassifierMixin:
    """Instantiate model type

    Parameters
    ----------
    model : str
        Name of the model (either logistic_regression or random_forest)

    Returns
    -------
    ClassifierMixin
        Instantiated classifier

    Raises
    ------
    KeyError
        Raises an error if the model type is not supported
    """
    if model in MODELS:
        Classifier, kwargs = MODELS[model]
        return Classifier(**kwargs)

    raise KeyError(
        f'The model {model} you specified is not available. Please use one of'
        ' {list(MODELS.keys())}')


def compute_metrics(df: pd.DataFrame) -> pd.Series:
    """Compute classification metrics

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing labels (ground truth), prediction, and score
        columns

    Returns
    -------
    pd.Series
        Contains accuracy, balanced accuracy, auroc and average precision
    """
    y_true = df['label']
    y_score = df['score']
    y_pred = df['prediction']

    return pd.Series({
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_score),
        'average_precision': average_precision_score(y_true, y_score)
    })


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('results_dir', type=click.Path())
@click.option('-m', '--model', type=click.STRING,
              default='logistic_regression',
              help='Specify the model')
def main(
    data_dir: Union[str, Path],
    results_dir: Union[str, Path],
    model: str
):
    """Run baseline method training

    Train a baseline model using preprocessed training data
    located in DATA_DIR. Saving all results to RESULTS_DIR.

    \f
    Parameters
    ----------
    data_dir : Union[str, Path]
        Location of preprocessed training data
    results_dir : Union[str, Path]
        Location for results
    model : str
        Name of the model type (logistic_regression or random_forest)
    """
    data_dir = Path(data_dir)

    x_train = np.load(data_dir / 'train.npy')
    x_valid = np.load(data_dir / 'valid.npy')

    df_train = pd.read_csv(data_dir / 'train.csv')
    df_valid = pd.read_csv(data_dir / 'valid.csv')

    y_train = df_train['label']
    y_valid = df_train['label']

    classifier = get_model(model)

    df_train['prediction'] = classifier.fit_predict(x_train, y_train)
    df_train['score'] = classifier.score(x_train)
    df_valid['prediction'] = classifier.predict(x_valid, y_valid)
    df_valid['score'] = classifier.score(x_valid)

    metric_train = compute_metrics(df_train)
    metric_train.name = 'train'
    metric_valid = compute_metrics(df_valid)
    metric_valid.name = 'valid'

    metrics = pd.concat([metric_train, metric_valid], axis=1).T
    logger.info(metrics)

    results_dir = Path(results_dir) / model
    results_dir.mkdir(exist_ok=True, parents=True)

    train_fn = results_dir / 'train.csv'
    df_train.to_csv(train_fn, index=False)
    valid_fn = results_dir / 'valid.csv'
    df_valid.to_csv(valid_fn, index=False)

    metrics_fn = results_dir / 'metrics.csv'
    metrics.to_csv(metrics_fn)

    model_fn = results_dir / 'model.joblib'
    dump(classifier, model_fn)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
