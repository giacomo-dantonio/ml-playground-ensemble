"""
Train an SVM model on the mnist dataset, downloaded using the fetch module.
"""

import argparse
import h5py
import joblib
import json
import logging
import numpy as np
import utils

from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

_logger = logging.getLogger(__name__)

_search_parameters = {
    "svm": {
        "clf__kernel": ["rbf"],
        "clf__C": [2, 4, 8, 10],
        "clf__gamma": ["scale", "auto"],
    },
    "forest": {
        "clf__n_estimators": [50, 100, 500],
        "clf__max_depth": [4, 16, 64],
        "clf__n_jobs": [-1]
    },
    "extra": {
        "clf__n_estimators": [50, 100, 500],
        "clf__max_depth": [4, 16, 64],
        "clf__n_jobs": [-1]
    },
    "sgd": {
        "clf__loss": ["hinge", "log"],
        "clf__penalty": ["l1", "l2"],
        "clf__alpha": [1e-4, 1e-2, 1e-1],
        "clf__n_jobs": [-1]
    },
}

def load(filepath):
    """
    Loads the MNIST dataset from an HDF5 file, as exported from the fetch module.

    Parameters:

    filepath    The path to the exported dataset.
    """

    _logger.info("Loading dataset from %s", filepath)
    with h5py.File(filepath, "r") as h5_file:
        return (
                np.array(h5_file["X_train"]),
                np.array(h5_file["y_train"]),
                np.array(h5_file["X_val"]),
                np.array(h5_file["y_val"]),
                np.array(h5_file["X_test"]),
                np.array(h5_file["y_test"]),
        )


def grid_search(X, y, alg="svm"):
    """
    Perform a grid search on a small validation set and returns
    the best model.

    Parameters:

    X       The features of the validation set
    y       The labels of the validation set
    alg     The algorithm used to train the model
    """

    _logger.info("Performing grid search.")
    with utils.Timer() as t:
        model = _make_pipeline(alg)

        f1_scorer = metrics.make_scorer(metrics.f1_score, greater_is_better=True, average='micro')
        search = model_selection.GridSearchCV(
            model,
            _search_parameters[alg],
            verbose=1,
            scoring=f1_scorer,
        )
        search.fit(X, y)

    _logger.info("Search time: {:.2f}s".format(t.elapsed))
    _logger.info("Search Accuracy: %s", search.best_score_)
    _logger.info("Best parameters: %s\n", json.dumps(search.best_params_, indent=2))

    return search.best_estimator_


def train(model, X, y):
    """
    Train a model on the training set.

    Parameters:

    model   The model to be trained.
    X       The features of the training set.
    y       The labels of the training set.
    """
    with utils.Timer() as t:
        model.fit(X, y)

    _logger.info("Training time: {:.2f}s".format(t.elapsed))


def _make_pipeline(model="svm", **kwargs):
    if model == "svm":
        clazz = svm.SVC
    elif model == "forest":
        clazz = ensemble.RandomForestClassifier
    elif model == "extra":
        clazz = ensemble.ExtraTreesClassifier
    elif model == "sgd":
        clazz = linear_model.SGDClassifier

    return pipeline.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("clf", clazz(**kwargs))
    ])


def _show_metrics(model, X_test, y_test):
    """
    Computes the classification metrics for the trained model,
    by applying it to the test set.
    """
    y_predict = model.predict(X_test)
    confusion = metrics.confusion_matrix(y_test, y_predict)
    report = metrics.classification_report(y_test, y_predict)

    return (
        "Metrics computed on the test set\n"
        "{report}\n"
        "Confusion matrix:\n{confusion}\n"
    ).format(
        report = report,
        confusion = confusion
    )


def _get_model_args(args):
    if args.model == "svm":
        try:
            gamma = float(args.gamma)
        except (TypeError, ValueError):
            gamma = args.gamma

        return {
            "C": args.regularization,
            "gamma": gamma
        }
    elif model in ["forest", "extra"]:
        return {
            "n_estimators": args.estimators,
            "max_depth": args.max_depth,
            "n_jobs": -1
        }
    elif model == "sgd":
        return {
            "loss": args.loss,
            "penalty": args.penalty,
            "alpha": args.alpha,
            "n_jobs": -1
        }

def _make_argparser() -> argparse.ArgumentParser:
    """Construct the parser for the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model from an HDF5 dataset and export it to a file.")

    parser.add_argument(
        "-g",
        "--gridsearch",
        action="store_true",
        help="Perform a grid search for optimizing the hyperparameters. "
            "If specified the model hyperparameters "
            "(e.g. --gamma and --regularization) are ignored.")

    parser.add_argument(
        "-d",
        "--dataset",
        default="data.h5",
        help="A filepath containing the dataset in the HDF5 format.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="model.pkl",
        help="The filepath for storing the trained model.")

    parser.add_argument(
        "-v",
        "---verbose",
        action="store_true",
        help="Display information about the trained model on stdout."
    )

    parser.add_argument(
        "-m",
        "--metrics",
        action="store_true",
        help="Display metrics computed on the test set."
    )

    parser.add_argument(
        "-V",
        "--validation",
        action="store_true",
        help="Use the validation set to compute the metrics."
    )

    parser.add_argument(
        "-C",
        "--regularization",
        type=int,
        default=8,
        help="Regularization parameter for SVM classifiers. "
            "The strength of the regularization is inversely proportional to C. "
            "Must be strictly positive. The penalty is a squared l2 penalty"
    )

    parser.add_argument(
        "-G",
        "--gamma",
        default="auto",
        help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. "
            "Can be a float value or one of 'scale' and 'auto'"
    )

    parser.add_argument(
        "-E",
        "--estimators",
        type=int,
        default=500,
        help="Number of estimators used for ensemble classifiers. "
             "Can be used with Random Forest or Extra Trees classifiers."
    )

    parser.add_argument(
        "-D",
        "--max_depth",
        type=int,
        default=16,
        help="Max depth of tree classifiers. "
             "Can be used with Random Forest or Extra Trees classifiers."
    )

    parser.add_argument(
        "-L",
        "--loss",
        default="hinge",
        help="The loss function to be used for SGD classifiers. "
             "Can be 'hinge', 'log', 'modified_huber', 'squared_hinge', "
             "'perceptron', or a regression loss: 'squared_error', 'huber', "
             "'epsilon_insensitive', or 'squared_epsilon_insensitive'."
    )

    parser.add_argument(
        "-P",
        "--penalty",
        default="l2",
        help="The penalty (aka regularization term) to be used for SGD classifiers. "
             "Can be either 'l1' or 'l2'."
    )

    parser.add_argument(
        "-A",
        "--alpha",
        type=float,
        default=1E-4,
        help="Constant that multiplies the regularization term for SGD classifiers. "
             "The higher the value, the stronger the regularization."
    )

    parser.add_argument(
        "-M",
        "--model",
        default="svm",
        help="The algorithm to be used for training the model. "
             "The following values are allowed: %s." % ", ".join(_search_parameters.keys())
    )

    return parser

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s\t%(message)s',
        level=logging.INFO if args.verbose else logging.WARNING
    )

    X_train, y_train, X_val, y_val, X_test, y_test = load(args.dataset)

    if args.gridsearch:
        model = grid_search(X_val, y_val, alg=args.model)
    else:
        model = _make_pipeline(_get_model_args(args))

    _logger.info("Training an %s classifier.", args.model)
    train(model, X_train, y_train)

    joblib.dump(model, args.outpath)

    if args.metrics:
        if args.validation:
            _logger.info(_show_metrics(model, X_val, y_val))
        else:
            _logger.info(_show_metrics(model, X_test, y_test))
