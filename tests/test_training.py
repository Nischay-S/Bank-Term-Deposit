"""Smoke/integration tests for the training pipeline, using a small SYNTHETIC frame in fast mode.
They check wiring, artifacts and leakage rules - never model quality."""
import json

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

import config
from src import data_loader as dl
from src import predict as pr
from src import train as tr
from tests.conftest import make_synthetic_frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("trained")
    df = make_synthetic_frame(1500, seed=21, positive_rate=0.2)
    # give the synthetic target some signal so every model/calibrator can fit
    df.loc[df["poutcome"] == "success", "y"] = "yes"
    orig = dl.load_raw
    dl.load_raw = lambda path=None: df
    try:
        meta = tr.train_scenario("pre_call", fast=True, out_dir=out)
    finally:
        dl.load_raw = orig
    return out / "pre_call", meta


def test_artifacts_are_written(trained):
    d, _ = trained
    for name in ("model.joblib", "base_model.joblib", "metadata.json", "model_comparison.csv", "cv_results.csv",
                 "calibration_comparison.csv", "threshold_sweep_val.csv", "test_predictions.csv"):
        assert (d / name).exists(), name


def test_metadata_records_reproducibility_info(trained):
    d, meta = trained
    on_disk = json.loads((d / "metadata.json").read_text())
    assert on_disk["training_config"]["seed"] == config.RANDOM_SEED
    for pkg in ("python", "scikit-learn", "numpy", "pandas", "imbalanced-learn", "shap"):
        assert pkg in on_disk["environment"]
    assert on_disk["scenario"] == "pre_call" and on_disk["target"] == config.TARGET
    assert 0 < on_disk["threshold"] < 1
    assert "validation" in on_disk["selection_metric"]


def test_pre_call_artifacts_exclude_duration(trained):
    _, meta = trained
    assert "duration" not in meta["input_features"] and meta["excluded_features"] == ["duration"]


def test_selected_model_is_best_on_validation_not_test(trained):
    d, meta = trained
    comp = pd.read_csv(d / "model_comparison.csv")
    val = comp[comp.split == "val"].set_index("model")
    assert meta["selected_model"] == val["pr_auc"].idxmax()


def test_every_model_scored_on_same_splits(trained):
    d, _ = trained
    comp = pd.read_csv(d / "model_comparison.csv")
    sizes = comp.groupby("split").apply(lambda g: (g["tp"] + g["fp"] + g["fn"] + g["tn"]).nunique(), include_groups=False)
    assert (sizes == 1).all()  # same number of rows for every model within a split


def test_saved_model_loads_and_predicts_from_raw_rows(trained):
    d, meta = trained
    bundle = pr.load_bundle("pre_call", artifact_dir=d)
    rows = make_synthetic_frame(20, seed=99).drop(columns=["y"])
    res, rep = pr.predict_frame(rows, bundle)
    assert res["probability"].between(0, 1).all() and rep.n_invalid == 0
    assert bundle.threshold == pytest.approx(meta["threshold"])


def test_test_predictions_are_probabilities_aligned_with_labels(trained):
    d, meta = trained
    tp = pd.read_csv(d / "test_predictions.csv", index_col=0)
    assert tp["proba"].between(0, 1).all() and set(tp["y_true"]) <= {0, 1}
    assert len(tp) == meta["training_config"]["n_test"]


def test_smote_lives_inside_the_pipeline_only():
    p = tr.base_candidates("pre_call")["logreg_smote"]
    assert isinstance(p, ImbPipeline) and "smote" in p.named_steps
    names = [n for n, _ in p.steps]
    assert names.index("smote") > names.index("encode")           # resampling happens after encoding, inside CV
    df = make_synthetic_frame(500, seed=4)
    X, y = dl.split_features_target(df)
    fitted = p.fit(X[config.scenario_features("pre_call")], y)
    out = fitted.predict_proba(X[config.scenario_features("pre_call")].head(7))
    assert out.shape == (7, 2)                                     # SMOTE is skipped at predict time


def test_baseline_and_imbalance_aware_candidates_exist():
    names = set(tr.base_candidates("pre_call"))
    assert "logreg_baseline" in names and {"logreg_balanced", "logreg_smote"} <= names
