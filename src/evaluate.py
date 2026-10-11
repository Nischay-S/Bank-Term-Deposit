"""Evaluation utilities: metrics, threshold analysis, calibration and bootstrap intervals."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def _arr(a):
    return np.asarray(a).ravel()


def expected_calibration_error(y, p, n_bins=10):
    """Weighted mean |observed frequency - mean predicted probability| over uniform bins."""
    y, p = _arr(y), _arr(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def confusion_counts(y, p, threshold):
    pred = (_arr(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(_arr(y), pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def full_metrics(y, p, threshold):
    """Threshold-free and threshold-based metrics for the positive (subscribed) class."""
    y, p = _arr(y), _arr(p)
    pred = (p >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
        "n_predicted_positive": int(pred.sum()),
        "prevalence": float(y.mean()),
    }
    out.update(confusion_counts(y, p, threshold))
    return out


def best_threshold_fbeta(y, p, beta=2.0):
    """Threshold maximising F-beta (computed exactly from the precision-recall curve)."""
    prec, rec, thr = precision_recall_curve(_arr(y), _arr(p))
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * prec * rec / (b2 * prec + rec)
    f = np.nan_to_num(f)
    i = int(np.argmax(f))
    return float(thr[i]), float(f[i])


def threshold_sweep(y, p, thresholds=None):
    """Precision / recall / F1 / number of predicted positives across thresholds."""
    y, p = _arr(y), _arr(p)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.99, 0.01), 2)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "n_predicted_positive": int(pred.sum()),
            "share_predicted_positive": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def class_report(y, p, threshold):
    pred = (_arr(p) >= threshold).astype(int)
    return classification_report(_arr(y), pred, target_names=["not_subscribed", "subscribed"],
                                 output_dict=True, zero_division=0)


def reliability_curve(y, p, n_bins=10):
    frac_pos, mean_pred = calibration_curve(_arr(y), _arr(p), n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def bootstrap_ci(y, p, threshold, n_rounds=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CIs over the test set for key metrics at a fixed threshold."""
    y, p = _arr(y), _arr(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = {k: [] for k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    for _ in range(n_rounds):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        pred = (pb >= threshold).astype(int)
        samples["roc_auc"].append(roc_auc_score(yb, pb))
        samples["pr_auc"].append(average_precision_score(yb, pb))
        samples["precision"].append(precision_score(yb, pred, zero_division=0))
        samples["recall"].append(recall_score(yb, pred, zero_division=0))
        samples["f1"].append(f1_score(yb, pred, zero_division=0))
    point = full_metrics(y, p, threshold)
    return {k: {"point": float(point[k]),
                "low": float(np.percentile(v, 100 * alpha / 2)),
                "high": float(np.percentile(v, 100 * (1 - alpha / 2)))}
            for k, v in samples.items()}
