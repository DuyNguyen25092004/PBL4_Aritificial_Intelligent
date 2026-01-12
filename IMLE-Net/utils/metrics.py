"""Metrics for evaluating the performance of a classification model.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score


def Metrics(y_true: NDArray, y_scores: NDArray) -> Tuple[list[float], float]:
    """Metrics for class-wise accuracy and mean accuracy.

    Parameters
    ----------
    y_true : NDArray
        Ground truth labels.
    y_scores : NDArray
        Predicted labels.

    Returns
    -------
    tuple[list[float], float]
        Tuple containing class-wise accuracy list and mean accuracy.

    """

    y_pred = y_scores >= 0.5
    acc = np.zeros(y_pred.shape[-1])

    for i in range(y_pred.shape[-1]):
        acc[i] = accuracy_score(y_true[:, i], y_pred[:, i])

    return acc.tolist(), float(np.mean(acc))


def AUC(y_true: NDArray, y_pred: NDArray, verbose: bool = False) -> list[float]:
    """Computes the macro-averaged AUC score.

    Parameters
    ----------
    y_true : NDArray
        Ground truth labels.
    y_pred : NDArray
        Predicted probabilities.
    verbose : bool, optional
        Whether to print verbose output. (default: False)

    Returns
    -------
    list[float]
        List of AUC scores per class.

    """

    aucs = []
    assert (
        len(y_true.shape) == 2 and len(y_pred.shape) == 2
    ), "Predictions and labels must be 2D."
    for col in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, col], y_pred[:, col]))
        except ValueError as e:
            if verbose:
                print(
                    f"Value error encountered for label {col}, likely due to using mixup or "
                    f"lack of full label presence. Setting AUC to accuracy. "
                    f"Original error was: {str(e)}."
                )
            aucs.append(float((y_pred == y_true).sum() / len(y_pred)))
    return aucs


def multi_threshold_precision_recall(
    y_true: NDArray, y_pred: NDArray, thresholds: NDArray
) -> Tuple[NDArray, NDArray]:
    """Precision and recall for different thresholds.

    Parameters
    ----------
    y_true : NDArray
        Ground truth labels.
    y_pred : NDArray
        Predicted probabilities.
    thresholds : NDArray
        Thresholds to use for computing precision and recall.

    Returns
    -------
    Tuple[NDArray, NDArray]
       Average precision and recall.

    """

    # Expand analysis to number of thresholds
    y_pred_bin = (
        np.repeat(y_pred[None, :, :], len(thresholds), axis=0)
        >= thresholds[:, None, None]
    )
    y_true_bin = np.repeat(y_true[None, :, :], len(thresholds), axis=0)

    # Compute true positives
    TP = np.sum(np.logical_and(y_true, y_pred_bin), axis=2)

    # Compute macro-average precision handling all warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sum(y_pred_bin, axis=2)
        precision = TP / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # Compute macro-average recall
    recall = TP / np.sum(y_true_bin, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def metric_summary(
    y_true: NDArray, y_pred: NDArray, num_thresholds: int = 10
) -> Tuple[float, float, list[float], list[float], list[float], list[float]]:
    """Metric summary for computing precision and recall at different thresholds. Also computes mean AUC and F1-scores

    Parameters
    ----------
    y_true : NDArray
        Ground truth labels.
    y_pred : NDArray
        Predicted probabilities.
    num_thresholds : int, optional
        Number of thresholds to use for computing precision and recall. (default: 10)

    Returns
    -------
    tuple[float, float, list[float], list[float], list[float], list[float]]
         Max F1 score, mean AUC, F1 scores, average precisions, average recalls, and thresholds.

    """

    thresholds = np.arange(0.00, 1.01, 1.0 / (num_thresholds - 1), float)
    average_precisions, average_recalls = multi_threshold_precision_recall(
        y_true, y_pred, thresholds
    )
    f_scores = (
        2
        * (average_precisions * average_recalls)
        / (average_precisions + average_recalls)
    )
    auc = float(np.array(AUC(y_true, y_pred, verbose=True)).mean())
    return (
        float(f_scores[np.nanargmax(f_scores)]),
        auc,
        f_scores.tolist(),
        average_precisions.tolist(),
        average_recalls.tolist(),
        thresholds.tolist(),
    )
