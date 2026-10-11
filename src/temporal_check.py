"""Supplementary sensitivity check: time-ordered split (NOT the main protocol).

The UCI documentation states that bank-full.csv is ordered by date (May 2008 - Nov 2010). The main study
uses a stratified RANDOM split (as the project brief instructs). If rows really are in time order, a random
split lets the model learn from the 'future'. This check trains on the first 60% of rows in FILE ORDER,
validates on the next 20% and tests on the last 20%, and compares with the random-split result.
File order is assumed to equal time order; the file has no year column, so this cannot be confirmed.
"""
import json
import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

import config
from src import data_loader as dl
from src import evaluate as ev
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


def month_runs(df):
    runs, prev = [], None
    for m in df["month"]:
        if m != prev:
            runs.append([m, 0])
            prev = m
        runs[-1][1] += 1
    return runs


def run(scenario="pre_call"):
    df = dl.load_raw()
    X, y = dl.split_features_target(df)
    feats = config.scenario_features(scenario)
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    parts = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}
    pipe = Pipeline([("prep", build_preprocessor(scenario, scale_numeric=False)),
                     ("clf", HistGradientBoostingClassifier(random_state=config.RANDOM_SEED))])
    pipe.fit(X.iloc[parts["train"]][feats], y.iloc[parts["train"]])
    pv = pipe.predict_proba(X.iloc[parts["val"]][feats])[:, 1]
    pt = pipe.predict_proba(X.iloc[parts["test"]][feats])[:, 1]
    yv, yt = y.iloc[parts["val"]].to_numpy(), y.iloc[parts["test"]].to_numpy()
    thr, _ = ev.best_threshold_fbeta(yv, pv, config.THRESHOLD_BETA)
    tm = ev.full_metrics(yt, pt, thr)
    runs = month_runs(df)
    meta = json.loads((config.ARTIFACT_DIR / scenario / "metadata.json").read_text())
    out = {
        "scenario": scenario, "assumption": "file order == time order (UCI documents date ordering)",
        "n_month_runs": len(runs), "month_runs_head": runs[:12],
        "prevalence": {k: float(y.iloc[s].mean()) for k, s in parts.items()},
        "months_in_test_block": sorted(set(df["month"].iloc[parts["test"]])),
        "months_in_train_block": sorted(set(df["month"].iloc[parts["train"]])),
        "temporal": {"val_pr_auc": float(average_precision_score(yv, pv)), "val_roc_auc": float(roc_auc_score(yv, pv)),
                     "test_pr_auc": tm["pr_auc"], "test_roc_auc": tm["roc_auc"], "threshold": thr,
                     "test_precision": tm["precision"], "test_recall": tm["recall"]},
        "random_split_reference": {"test_pr_auc": meta["metrics_test"]["pr_auc"], "test_roc_auc": meta["metrics_test"]["roc_auc"],
                                   "selected_model": meta["selected_model"]},
        "note": "Model here is default HistGradientBoosting trained without calibration; the reference is the final calibrated model.",
    }
    (config.ARTIFACT_DIR / scenario / "temporal_check.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(), indent=2))
