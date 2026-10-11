"""Shared fixtures. The synthetic frame is a TEST FIXTURE only; no metrics come from it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def make_synthetic_frame(n=400, seed=0, positive_rate=0.15):
    rng = np.random.default_rng(seed)
    pdays = np.where(rng.random(n) < 0.8, -1, rng.integers(1, 400, n))
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "job": rng.choice(["admin.", "technician", "management", "unknown"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
        "default": rng.choice(["yes", "no"], n, p=[0.02, 0.98]),
        "balance": rng.integers(-500, 20000, n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "day": rng.integers(1, 29, n),
        "month": rng.choice(["jan", "may", "aug", "nov"], n),
        "duration": rng.integers(0, 1500, n),
        "campaign": rng.integers(1, 10, n),
        "pdays": pdays,
        "previous": np.where(pdays == -1, 0, rng.integers(1, 5, n)),
        "poutcome": rng.choice(["failure", "other", "success", "unknown"], n),
    })
    y = np.where(rng.random(n) < positive_rate, "yes", "no")
    y[:5] = "yes"; y[5:10] = "no"  # guarantee both classes
    df["y"] = y
    return df


@pytest.fixture
def synthetic_df():
    return make_synthetic_frame()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    p = tmp_path / "bank.csv"
    synthetic_df.to_csv(p, index=False)
    return p


@pytest.fixture
def real_data_path():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("Real dataset not present at data/raw/bank_marketing.csv")
    return config.RAW_DATA_PATH


# ---- fake (synthetic) trained artifacts, used only to test prediction/explanation LOGIC ----
import json  # noqa: E402

import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.preprocessing import build_preprocessor  # noqa: E402


def make_fake_artifacts(root, scenario, df, threshold=0.3):
    X = df[config.scenario_features(scenario)]
    y = (df["y"] == "yes").astype(int)
    pipe = Pipeline([("prep", build_preprocessor(scenario)),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, root / "model.joblib")
    joblib.dump(pipe, root / "base_model.joblib")
    (root / "metadata.json").write_text(json.dumps({
        "scenario": scenario, "input_features": config.scenario_features(scenario),
        "threshold": threshold, "selected_model": "fake_logreg", "calibration_method": "none"}))
    return root


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_artifacts")
    df = make_synthetic_frame(600, seed=3)
    for sc in config.SCENARIOS:
        make_fake_artifacts(root / sc, sc, df)
    return root


@pytest.fixture(scope="session")
def pre_bundle(fake_root):
    from src.predict import load_bundle
    return load_bundle("pre_call", artifact_dir=fake_root / "pre_call")


@pytest.fixture(scope="session")
def post_bundle(fake_root):
    from src.predict import load_bundle
    return load_bundle("post_call", artifact_dir=fake_root / "post_call")


@pytest.fixture
def customers():
    return make_synthetic_frame(30, seed=11).drop(columns=["y"])
