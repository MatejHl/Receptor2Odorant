import io
import itertools
from functools import partial
import warnings
import numpy as np
from matplotlib import pyplot as plt
from objax.jaxboard import Image
import jax
from jax import numpy as jnp
from jax.experimental import sparse as jsp
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics._plot.roc_curve import RocCurveDisplay
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay
from sklearn.metrics._classification import _prf_divide, _warn_prf
from sklearn.metrics._base import _average_binary_score
from sklearn.metrics._ranking import auc


def confusion_matrix_binary(y_true, y_pred, labels, sample_weight=None):
    """
    Jittable veriosn of sklearn.confusion_matrix assuming binary classification.

    Extra inputs that are not used for the purpose of this model are omitted so that we have pure function.

    References:
    -----------
    https://github.com/scikit-learn/scikit-learn/blob/582fa30a3/sklearn/metrics/_classification.py#L222
    """
    n_labels = labels.size

    if n_labels == 0:
        raise ValueError("'labels' should contains at least one label.")
    elif y_true.size == 0:
        return jnp.zeros((n_labels, n_labels), dtype=jnp.int32)

    if sample_weight is None:
        sample_weight = jnp.ones(y_true.shape[0], dtype=jnp.int32)

    cm = jsp.BCOO((sample_weight, jnp.stack([y_true, y_pred], axis = -1)),
        shape =(n_labels, n_labels)
    ).todense()
    return cm


def plot_confusion_matrix(cm, class_names):
    viz = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_names)
    return viz.plot().figure_


def plot_to_png(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a numpy array and
    returns it.
    """
    buf = io.BytesIO()    
    plt.savefig(buf, format='png')
    shape = (int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), 4)

    plt.close(figure)
    buf.seek(0)

    image = buf.getvalue()
    return image, shape


def log_confusion_matrix(cm, class_names):
    figure = plot_confusion_matrix(cm, class_names)
    image, shape = plot_to_png(figure)
    return Image(shape, image)


def precision_recall_fscore(cm, beta=1.0, warn_for=("precision", "recall", "f-score"), zero_division="warn"):
    """
    Reference:
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/metrics/_classification.py#L1380
    """
    average = None
    tp_sum = np.array([cm[1,1]])
    pred_sum = tp_sum + np.array([cm[0,1]])
    true_sum = tp_sum + np.array([cm[1,0]])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division)
    recall = _prf_divide(tp_sum, true_sum, "recall", "true", average, warn_for, zero_division)

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    return precision, recall, f_score


def MCC(cm):
    """
    Reference:
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/metrics/_classification.py#L829
    """
    C = cm
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


def true_negative_rate(cm):
    tn = cm[0,0]
    n = cm[0,1] + cm[0,0]
    return tn/n


def fp_tp_per_threshold(cm):
    """
    """
    tp = cm[1,1]
    # fn = cm[1,0]
    fp = cm[0,1]
    # tn = cm[0,0]

    return fp, tp


def precision_recall_curve(cm_per_threshold):
    """
    get values of precesion-recall curve. This is based on sklearn implementation of PR curve.

    Reference:
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/32f9deaaf/sklearn/metrics/_ranking.py#L786
    """
    # NOTE: Rest of the code assumes fp and tp to be ascending.
    #       thresholds are ascnding and needs to be reversed for fp and tp to be ascending.
    vals = np.array([fp_tp_per_threshold(cm_per_threshold[key]) for key in reversed(cm_per_threshold.keys())])
    fps=vals[:, 0]
    tps=vals[:, 1]
    # thresholds = np.array(list(cm_per_threshold.keys()))
    thresholds = np.fromiter(cm_per_threshold.keys(), dtype=float)

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def average_precision_score(pr_curve_values):
    """
    Based on sklearn implementation.

    References:
    -----------
    https://github.com/scikit-learn/scikit-learn/blob/32f9deaaf/sklearn/metrics/_ranking.py#L111
    """
    precision, recall, _ = pr_curve_values
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def plot_pr_curve(pr_curve_values, average_precision = None):
    """
    Returns a matplotlib figure containing the plotted ROC curve.
    
    Paramters:
    ----------
    roc_curve_values : tuple
        result of roc_curve. tuple of (tpr, fpr, thresholds)
    """
    precision, recall, _ = pr_curve_values
    viz = PrecisionRecallDisplay(precision, recall, average_precision = average_precision)
    return viz.plot().figure_

def log_pr_curve(pr_curve_values, average_precision = None):
    figure = plot_pr_curve(pr_curve_values, average_precision = average_precision)
    image, shape = plot_to_png(figure)
    return Image(shape, image)


def roc_auc(roc_curve_values, max_fpr=None):
    """
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/metrics/_ranking.py#L47
    """
    fpr, tpr, _ = roc_curve_values

    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected max_fpr in range (0, 1], got: %r" % max_fpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    # McClish correction: standardize result to be 0.5 if non-discriminant
    # and 1 if maximal
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def roc_curve(cm_per_threshold, drop_intermediate=True):
    """
    get false positive rate and true positive rate for all thresholds. This is based on sklearn implementation of ROC curve.

    References:
    -----------
    https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/metrics/_ranking.py#L873
    """
    # NOTE: Rest of the code assumes fp and tp to be ascending.
    #       thresholds are ascnding and needs to be reversed for fp and tp to be ascending.
    vals = np.array([fp_tp_per_threshold(cm_per_threshold[key]) for key in reversed(cm_per_threshold.keys())])
    fps=vals[:, 0]
    tps=vals[:, 1]
    # thresholds = np.array(list(cm_per_threshold.keys()))
    thresholds = np.fromiter(cm_per_threshold.keys(), dtype=float)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0: # TODO: This is assuming we do not have a perfect classifier?
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0: # TODO: This is assuming we do not have a worst possible classifier?
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless",
            UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1] 

    return fpr, tpr, thresholds


def plot_roc_curve(roc_curve_values):
    """
    Returns a matplotlib figure containing the plotted ROC curve.
    
    Paramters:
    ----------
    roc_curve_values : tuple
        result of roc_curve. tuple of (tpr, fpr, thresholds)
    """
    fpr, tpr, _ = roc_curve_values

    viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=None, estimator_name=None, pos_label=None)
    return viz.plot().figure_


def log_roc_curve(roc_curve_values):
    figure = plot_roc_curve(roc_curve_values)
    image, shape = plot_to_png(figure)
    return Image(shape, image)