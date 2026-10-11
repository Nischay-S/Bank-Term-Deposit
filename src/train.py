"""Training pipeline: CV comparison, light tuning, calibration, threshold selection, artifacts.

Usage:
    python -m src.train --scenario pre_call
    python -m src.train --scenario all [--fast]
Model selection and threshold choice use ONLY train/validation data. The test split is scored
after selection, for reporting.
"""
import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from src import data_loader as dl
from src import evaluate as ev
from src.preprocessing import build_preprocessor

logger = logging.getLogger("train")


# ---------------------------------------------------------------- candidates
def _pipe(scenario, clf, scale=True, smote=False):
    prep = build_preprocessor(scenario, scale_numeric=scale)
    if smote:
        # imblearn forbids nested Pipelines as intermediate steps, so the preprocessing steps are flattened.
        return ImbPipeline(list(prep.steps) + [("smote", SMOTE(random_state=config.RANDOM_SEED)), ("clf", clf)])
    return Pipeline([("prep", prep), ("clf", clf)])


def base_candidates(scenario, seed=config.RANDOM_SEED):
    lr = lambda **kw: LogisticRegression(max_iter=2000, random_state=seed, **kw)  # noqa: E731
    return {
        "logreg_baseline": _pipe(scenario, lr()),
        "logreg_balanced": _pipe(scenario, lr(class_weight="balanced")),
        "logreg_smote": _pipe(scenario, lr(), smote=True),
        "decision_tree_balanced": _pipe(scenario, DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=seed), scale=False),
        "random_forest_balanced": _pipe(scenario, RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1), scale=False),
        "hist_gradient_boosting": _pipe(scenario, HistGradientBoostingClassifier(random_state=seed), scale=False),
    }


SEARCH_SPACES = {
    "random_forest_balanced": {
        "clf__max_depth": [8, 12, 16, 24], "clf__min_samples_leaf": [3, 5, 10, 20],
        "clf__max_features": ["sqrt", 0.3],
    },
    "hist_gradient_boosting": {
        "clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_iter": [150, 300],
        "clf__max_leaf_nodes": [15, 31, 63], "clf__min_samples_leaf": [20, 50, 100],
        "clf__l2_regularization": [0.0, 1.0, 10.0], "clf__class_weight": [None, "balanced"],
    },
}


def tune(name, pipe, X, y, n_iter, seed):
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACES[name], n_iter=n_iter, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), random_state=seed, n_jobs=1, refit=False)
    search.fit(X, y)
    best = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    tuned = clone(pipe).set_params(**search.best_params_)
    logger.info("Tuned %s: best CV PR-AUC=%.4f params=%s", name, search.best_score_, best)
    return tuned, best, float(search.best_score_)


# ---------------------------------------------------------------- helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def environment_info():
    import shap
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__, "joblib": joblib.__version__, "shap": shap.__version__,
    }


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- main routine
def train_scenario(scenario, fast=False, out_dir=None):
    t0 = time.time()
    seed = config.RANDOM_SEED
    out = (out_dir or config.ARTIFACT_DIR) / scenario
    out.mkdir(parents=True, exist_ok=True)

    df = dl.load_raw()
    X, y = dl.split_features_target(df)
    s = dl.make_splits(X, y, seed=seed)
    feats = config.scenario_features(scenario)
    Xtr, Xva, Xte = s.X_train[feats], s.X_val[feats], s.X_test[feats]
    logger.info("[%s] train=%d val=%d test=%d | positive rate train=%.4f val=%.4f test=%.4f",
                scenario, len(Xtr), len(Xva), len(Xte), s.y_train.mean(), s.y_val.mean(), s.y_test.mean())

    # 1) candidates (+ tuned versions of the two strongest tree families)
    cands = base_candidates(scenario, seed)
    n_iter = 3 if fast else 5
    tuning_info = {}
    for name in ("random_forest_balanced", "hist_gradient_boosting"):
        tuned, best, score = tune(name, cands[name], Xtr, s.y_train, n_iter, seed)
        key = name.replace("_balanced", "") + "_tuned"
        cands[key] = tuned
        tuning_info[key] = {"best_params": best, "inner_cv_pr_auc": score, "n_iter": n_iter}

    # 2) cross-validation on the TRAINING split only
    cv = StratifiedKFold(CV_FOLDS := (3 if fast else config.CV_FOLDS), shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall",
               "precision": "precision", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}
    cv_rows = []
    for name, pipe in cands.items():
        res = cross_validate(pipe, Xtr, s.y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            row[f"{m}_mean"] = float(np.mean(res[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(res[f"test_{m}"]))
        cv_rows.append(row)
        logger.info("[%s] CV %-24s PR-AUC=%.4f ROC-AUC=%.4f", scenario, name, row["pr_auc_mean"], row["roc_auc_mean"])
    cv_df = pd.DataFrame(cv_rows)

    # 3) fit on train, evaluate on validation (selection) and test (reporting, same split for all)
    fitted, comp_rows = {}, []
    for name, pipe in cands.items():
        m = clone(pipe).fit(Xtr, s.y_train)
        fitted[name] = m
        pv, pt = _proba(m, Xva), _proba(m, Xte)
        thr, _ = ev.best_threshold_fbeta(s.y_val, pv, config.THRESHOLD_BETA)
        for split, yy, pp in (("val", s.y_val, pv), ("test", s.y_test, pt)):
            r = ev.full_metrics(yy, pp, thr)
            comp_rows.append({"model": name, "split": split, **r})
    comp = pd.DataFrame(comp_rows)

    # 4) select on VALIDATION PR-AUC
    val = comp[comp.split == "val"].set_index("model")
    selected = str(val[config.SELECTION_METRIC].idxmax())
    logger.info("[%s] Selected by validation %s: %s (%.4f)", scenario, config.SELECTION_METRIC,
                selected, val.loc[selected, config.SELECTION_METRIC])
    base = fitted[selected]

    # 5) calibration: none vs sigmoid vs isotonic (fit on train via internal CV; judged on validation Brier)
    pv_base = _proba(base, Xva)
    cal_rows = [{"method": "none", **{k: ev.full_metrics(s.y_val, pv_base, 0.5)[k] for k in ("brier", "log_loss", "ece", "pr_auc", "roc_auc")}}]
    cal_models = {"none": base}
    for method in ("sigmoid", "isotonic"):
        cm = CalibratedClassifierCV(estimator=clone(cands[selected]), method=method, cv=5).fit(Xtr, s.y_train)
        cal_models[method] = cm
        r = ev.full_metrics(s.y_val, _proba(cm, Xva), 0.5)
        cal_rows.append({"method": method, **{k: r[k] for k in ("brier", "log_loss", "ece", "pr_auc", "roc_auc")}})
    cal_df = pd.DataFrame(cal_rows)
    best_cal = str(cal_df.loc[cal_df.brier.idxmin(), "method"])
    final = cal_models[best_cal]
    logger.info("[%s] Calibration choice by validation Brier: %s", scenario, best_cal)

    # 6) threshold on validation probabilities of the FINAL (calibrated) model
    pv = _proba(final, Xva)
    thr, fbeta_val = ev.best_threshold_fbeta(s.y_val, pv, config.THRESHOLD_BETA)

    # 7) final evaluation on the held-out test split
    pt = _proba(final, Xte)
    test_metrics = ev.full_metrics(s.y_test, pt, thr)
    val_metrics = ev.full_metrics(s.y_val, pv, thr)
    ci = ev.bootstrap_ci(s.y_test, pt, thr, n_rounds=200 if fast else config.BOOTSTRAP_ROUNDS, seed=seed)
    report = ev.class_report(s.y_test, pt, thr)

    # 8) save artifacts
    joblib.dump(final, out / "model.joblib")
    joblib.dump(base, out / "base_model.joblib")  # uncalibrated; used for SHAP explanations
    comp.to_csv(out / "model_comparison.csv", index=False)
    cv_df.to_csv(out / "cv_results.csv", index=False)
    cal_df.to_csv(out / "calibration_comparison.csv", index=False)
    ev.threshold_sweep(s.y_val, pv).to_csv(out / "threshold_sweep_val.csv", index=False)
    ev.threshold_sweep(s.y_test, pt).to_csv(out / "threshold_sweep_test.csv", index=False)
    ev.reliability_curve(s.y_test, pt).to_csv(out / "reliability_final_test.csv", index=False)
    ev.reliability_curve(s.y_test, _proba(base, Xte)).to_csv(out / "reliability_uncalibrated_test.csv", index=False)
    pd.DataFrame({"y_true": s.y_test.values, "proba": pt}, index=s.y_test.index).to_csv(out / "test_predictions.csv")
    pd.DataFrame({"y_true": s.y_val.values, "proba": pv}, index=s.y_val.index).to_csv(out / "val_predictions.csv")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario, "target": config.TARGET, "positive_label": config.POSITIVE_LABEL,
        "input_features": feats, "excluded_features": config.LEAKAGE_FEATURES[scenario],
        "selected_model": selected, "calibration_method": best_cal,
        "selection_metric": f"validation {config.SELECTION_METRIC}",
        "threshold": thr, "threshold_objective": f"maximise F{config.THRESHOLD_BETA:g} on validation data",
        "validation_fbeta_at_threshold": fbeta_val,
        "metrics_validation": val_metrics, "metrics_test": test_metrics, "test_bootstrap_ci_95": ci,
        "classification_report_test": report,
        "tuning": tuning_info,
        "training_config": {"seed": seed, "cv_folds": CV_FOLDS, "val_size": config.VAL_SIZE,
                            "test_size": config.TEST_SIZE, "fast_mode": fast, "n_train": len(Xtr),
                            "n_val": len(Xva), "n_test": len(Xte)},
        "environment": environment_info(),
        "train_seconds": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(_jsonable(meta), indent=2))
    logger.info("[%s] Done in %.0fs. Artifacts in %s", scenario, time.time() - t0, out)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Train BankSmart AI models")
    ap.add_argument("--scenario", choices=list(config.SCENARIOS) + ["all"], default="all")
    ap.add_argument("--fast", action="store_true", help="smoke-test settings (less tuning, fewer folds)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for sc in (config.SCENARIOS if args.scenario == "all" else [args.scenario]):
        train_scenario(sc, fast=args.fast)


if __name__ == "__main__":
    main()
