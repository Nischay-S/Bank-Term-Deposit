import numpy as np
import pandas as pd
import pytest

import config
from src import data_loader as dl
from src import predict as pr
from src.preprocessing import FeatureMismatchError


# ---------------- artifacts ----------------
def test_missing_artifacts_fail_gracefully(tmp_path):
    with pytest.raises(pr.ArtifactsMissingError, match="python -m src.train"):
        pr.load_bundle("pre_call", artifact_dir=tmp_path)


def test_scenario_mismatch_rejected(fake_root):
    with pytest.raises(pr.ArtifactsMissingError, match="trained for scenario"):
        pr.load_bundle("post_call", artifact_dir=fake_root / "pre_call")


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        pr.load_bundle("sometime")


def test_pre_and_post_artifacts_are_separate(pre_bundle, post_bundle, fake_root):
    assert "duration" not in pre_bundle.input_features
    assert "duration" in post_bundle.input_features
    assert pre_bundle.metadata["scenario"] == "pre_call" and post_bundle.metadata["scenario"] == "post_call"
    assert (fake_root / "pre_call").resolve() != (fake_root / "post_call").resolve()


# ---------------- single prediction ----------------
def test_single_prediction_probability_in_range(pre_bundle, customers):
    out = pr.predict_single(customers.iloc[0].to_dict(), pre_bundle)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["predicted_class"] in (0, 1) and out["predicted_label"] in ("yes", "no")


def test_threshold_controls_class(pre_bundle, customers):
    row = customers.iloc[0].to_dict()
    assert pr.predict_single(row, pre_bundle, threshold=0.0)["predicted_class"] == 1
    assert pr.predict_single(row, pre_bundle, threshold=1.0)["predicted_class"] == 0


@pytest.mark.parametrize("bad", [1.5, -0.1, "high", None.__class__])
def test_invalid_threshold_rejected(pre_bundle, customers, bad):
    with pytest.raises(pr.InputValidationError):
        pr.predict_single(customers.iloc[0].to_dict(), pre_bundle, threshold=bad)


def test_single_invalid_value_raises(pre_bundle, customers):
    row = customers.iloc[0].to_dict(); row["age"] = 5
    with pytest.raises(pr.InputValidationError, match="age"):
        pr.predict_single(row, pre_bundle)


# ---------------- batch prediction ----------------
def test_batch_prediction_shape_range_and_consistency(pre_bundle, customers):
    res, rep = pr.predict_frame(customers, pre_bundle)
    assert len(res) == len(customers) and (res["status"] == "ok").all()
    assert res["probability"].between(0, 1).all()
    thr = pre_bundle.threshold
    assert ((res["probability"] >= thr).astype(int) == res["predicted_class"].astype(int)).all()
    assert rep.n_invalid == 0


def test_higher_threshold_never_adds_positives(pre_bundle, customers):
    lo, _ = pr.predict_frame(customers, pre_bundle, threshold=0.2)
    hi, _ = pr.predict_frame(customers, pre_bundle, threshold=0.6)
    assert hi["predicted_class"].astype(int).sum() <= lo["predicted_class"].astype(int).sum()


def test_missing_required_column_gives_helpful_error(pre_bundle, customers):
    with pytest.raises(FeatureMismatchError, match="balance"):
        pr.predict_frame(customers.drop(columns=["balance"]), pre_bundle)


def test_near_match_column_suggested_but_not_applied(pre_bundle, customers):
    with pytest.raises(FeatureMismatchError, match="NOT applied"):
        pr.predict_frame(customers.rename(columns={"balance": "balanace"}), pre_bundle)


def test_pre_call_ignores_and_reports_duration(pre_bundle, customers):
    with_dur, rep = pr.predict_frame(customers, pre_bundle)          # customers include duration
    without, _ = pr.predict_frame(customers.drop(columns=["duration"]), pre_bundle)
    assert rep.leakage_columns_ignored == ["duration"]
    np.testing.assert_allclose(with_dur["probability"], without["probability"])


def test_changing_duration_does_not_change_pre_call_prediction(pre_bundle, customers):
    a, _ = pr.predict_frame(customers, pre_bundle)
    b, _ = pr.predict_frame(customers.assign(duration=9999), pre_bundle)
    np.testing.assert_allclose(a["probability"], b["probability"])


def test_post_call_requires_duration_and_uses_it(post_bundle, customers):
    with pytest.raises(FeatureMismatchError, match="duration"):
        pr.predict_frame(customers.drop(columns=["duration"]), post_bundle)
    a, _ = pr.predict_frame(customers, post_bundle)
    b, _ = pr.predict_frame(customers.assign(duration=5000), post_bundle)
    assert not np.allclose(a["probability"], b["probability"])


def test_unseen_category_flagged_but_predicted(pre_bundle, customers):
    df = customers.copy(); df.loc[df.index[0], "job"] = "astronaut"
    res, rep = pr.predict_frame(df, pre_bundle)
    assert res.iloc[0]["status"] == "warning" and "astronaut" in res.iloc[0]["data_quality_note"]
    assert not np.isnan(res.iloc[0]["probability"])
    assert rep.unseen_categories == {"job": ["astronaut"]}


def test_missing_values_flagged_but_predicted(pre_bundle, customers):
    df = customers.copy(); df.loc[df.index[0], "balance"] = np.nan; df.loc[df.index[1], "job"] = None
    res, rep = pr.predict_frame(df, pre_bundle)
    assert (res.iloc[:2]["status"] == "warning").all() and res.iloc[:2]["probability"].notna().all()
    assert rep.missing_cells == 2


def test_invalid_rows_get_no_prediction_and_others_still_work(pre_bundle, customers):
    df = customers.copy().astype({"age": object})
    df.loc[df.index[0], "age"] = "abc"          # non-numeric
    df.loc[df.index[1], "age"] = 3               # out of range
    res, rep = pr.predict_frame(df, pre_bundle)
    assert (res.iloc[:2]["status"] == "invalid").all()
    assert res.iloc[:2]["probability"].isna().all() and res.iloc[:2]["predicted_label"].isna().all()
    assert (res.iloc[2:]["status"] == "ok").all() and res.iloc[2:]["probability"].notna().all()
    assert rep.n_invalid == 2


def test_empty_input_rejected(pre_bundle, customers):
    with pytest.raises(pr.InputValidationError):
        pr.predict_frame(customers.head(0), pre_bundle)


# ---------------- real trained artifacts (skipped if not trained) ----------------
@pytest.fixture(scope="module")
def real_bundles():
    try:
        return {sc: pr.load_bundle(sc) for sc in config.SCENARIOS}
    except pr.ArtifactsMissingError:
        pytest.skip("Real artifacts not trained")


def test_real_artifacts_reproduce_saved_test_predictions(real_bundles, real_data_path):
    X, y = dl.split_features_target(dl.load_raw(real_data_path))
    s = dl.make_splits(X, y)
    for sc, b in real_bundles.items():
        saved = pd.read_csv(config.ARTIFACT_DIR / sc / "test_predictions.csv", index_col=0)
        res, _ = pr.predict_frame(s.X_test, b)
        assert (res.index == saved.index).all()
        np.testing.assert_allclose(res["probability"].to_numpy(), saved["proba"].to_numpy(), atol=1e-9)
        assert 0 < b.threshold < 1


def test_real_pre_call_artifact_excludes_duration(real_bundles):
    assert "duration" not in real_bundles["pre_call"].input_features
    assert "duration" in real_bundles["post_call"].input_features
    assert real_bundles["pre_call"].metadata["excluded_features"] == ["duration"]
