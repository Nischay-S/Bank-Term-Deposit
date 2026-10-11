"""Feature-ablation study (pre-call scenario): does removing a feature group change test performance?

Uses the selected architecture (HistGradientBoosting, default hyper-parameters), refit on the training
split for each variant. Reports threshold-free metrics, with a paired bootstrap 95% CI for the change in
PR-AUC versus the full model on the same test rows. Differences whose CI includes 0 are not distinguishable
from sampling noise.
"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

import config
from src import data_loader as dl
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

VARIANTS = {
    "all pre-call features": (),
    "without campaign": ("campaign",),
    "without day, month": ("day", "month"),
    "without previous, poutcome": ("previous", "poutcome"),
    "without age, marital (sensitive attributes)": ("age", "marital"),
}


def run(scenario="pre_call", n_boot=500, seed=config.RANDOM_SEED):
    X, y = dl.split_features_target(dl.load_raw())
    s = dl.make_splits(X, y, seed=seed)
    feats = config.scenario_features(scenario)
    preds, rows = {}, []
    for name, excl in VARIANTS.items():
        pipe = Pipeline([("prep", build_preprocessor(scenario, scale_numeric=False, exclude=excl)),
                         ("clf", HistGradientBoostingClassifier(random_state=seed))])
        pipe.fit(s.X_train[feats], s.y_train)
        pv, pt = pipe.predict_proba(s.X_val[feats])[:, 1], pipe.predict_proba(s.X_test[feats])[:, 1]
        preds[name] = pt
        rows.append({"variant": name, "removed": ", ".join(excl) or "-",
                     "val_pr_auc": average_precision_score(s.y_val, pv), "val_roc_auc": roc_auc_score(s.y_val, pv),
                     "test_pr_auc": average_precision_score(s.y_test, pt), "test_roc_auc": roc_auc_score(s.y_test, pt)})
        logger.info("ablation %-45s val PR-AUC=%.4f test PR-AUC=%.4f", name, rows[-1]["val_pr_auc"], rows[-1]["test_pr_auc"])
    out = pd.DataFrame(rows)
    yt, rng, n = s.y_test.to_numpy(), np.random.default_rng(seed), len(s.y_test)
    full = preds["all pre-call features"]
    d_lo, d_hi = [0.0], [0.0]
    for name in list(VARIANTS)[1:]:
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            diffs.append(average_precision_score(yt[i], preds[name][i]) - average_precision_score(yt[i], full[i]))
        d_lo.append(float(np.percentile(diffs, 2.5))); d_hi.append(float(np.percentile(diffs, 97.5)))
    out["delta_test_pr_auc"] = out["test_pr_auc"] - out["test_pr_auc"].iloc[0]
    out["delta_ci_low"], out["delta_ci_high"] = d_lo, d_hi
    out.to_csv(config.ARTIFACT_DIR / scenario / "ablation.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(); ap.parse_args()
    pd.set_option("display.width", 250)
    print(run().round(4).to_string(index=False))
