import numpy as np
import pandas as pd
import pytest

import config
from src import campaign as cp
from src import data_loader as dl
from src import explain as ex
from src import predict as pr
from tests.conftest import make_synthetic_frame


# ---------------- campaign ----------------
def _toy():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(0.95, 0.05, 10)
    return y, p


def test_campaign_table_hand_computed():
    y, p = _toy()
    t = cp.campaign_table(y, p, fractions=(0.2, 0.5)).set_index("outreach_fraction")
    assert t.loc[0.2, "customers_selected"] == 2 and t.loc[0.2, "subscribers_found"] == 2
    assert t.loc[0.2, "precision"] == 1.0 and t.loc[0.2, "recall"] == 1.0 and t.loc[0.2, "lift"] == pytest.approx(5.0)
    assert t.loc[0.5, "precision"] == pytest.approx(0.4) and t.loc[0.5, "lift"] == pytest.approx(2.0)
    assert t.loc[0.5, "random_expected_subscribers"] == pytest.approx(1.0)


def test_campaign_full_outreach_has_lift_one_and_full_recall():
    y, p = _toy()
    t = cp.campaign_table(y, p, fractions=(1.0,)).iloc[0]
    assert t.lift == pytest.approx(1.0) and t.recall == 1.0


@pytest.mark.parametrize("f", [0, -0.1, 1.5])
def test_campaign_rejects_bad_fraction(f):
    y, p = _toy()
    with pytest.raises(ValueError):
        cp.campaign_table(y, p, fractions=(f,))


def test_campaign_requires_positives():
    with pytest.raises(ValueError):
        cp.campaign_table(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_gains_curve_monotone_and_complete():
    y, p = _toy()
    g = cp.gains_curve(y, p, n_points=10)
    assert (np.diff(g["recall"]) >= 0).all() and g["recall"].iloc[-1] == 1.0


def test_select_top_fraction():
    df = pd.DataFrame({"probability": [0.1, 0.9, 0.5, 0.7]})
    top = cp.select_top_fraction(df, 0.5)
    assert sorted(top["probability"]) == [0.7, 0.9]


def test_real_campaign_lift_exceeds_one_when_artifacts_exist():
    path = config.ARTIFACT_DIR / "pre_call" / "test_predictions.csv"
    if not path.exists():
        pytest.skip("artifacts not trained")
    preds = pd.read_csv(path, index_col=0)
    t = cp.campaign_table(preds.y_true, preds.proba)
    assert (t["lift"] > 1).all() and (np.diff(t["recall"]) > 0).all()
    assert (t["subscribers_found"] <= t["customers_selected"]).all()


# ---------------- explanations ----------------
@pytest.fixture(scope="module")
def background():
    return make_synthetic_frame(300, seed=5).drop(columns=["y"])


def test_explainer_local_explanation_pre_call(pre_bundle, background):
    e = ex.ModelExplainer(pre_bundle, background)
    out = e.explain_row(background.head(1))
    assert e.method in ("shap_linear", "shap_tree")
    assert "duration" not in set(out["feature"])
    assert {"feature", "value", "contribution"} <= set(out.columns)


def test_explainer_post_call_includes_duration(post_bundle, background):
    out = ex.ModelExplainer(post_bundle, background).explain_row(background.head(1))
    assert "duration" in set(out["feature"])


def test_linear_shap_is_additive_in_log_odds(pre_bundle, background):
    e = ex.ModelExplainer(pre_bundle, background)
    row = background.head(1)
    out = e.explain_row(row)
    p = pre_bundle.base_model.predict_proba(row)[:, 1][0]
    recon = 1 / (1 + np.exp(-(out.attrs["base_value"] + out["contribution"].sum())))
    assert recon == pytest.approx(p, abs=1e-6)


def test_occlusion_fallback_works(pre_bundle, background):
    e = ex.ModelExplainer(pre_bundle, background, method="occlusion")
    out = e.explain_row(background.head(1))
    assert e.method == "occlusion" and set(out["feature"]) == set(pre_bundle.input_features)


def test_shap_failure_falls_back_without_crashing(pre_bundle, background, monkeypatch):
    import shap
    def boom(*a, **k):
        raise RuntimeError("incompatible")
    monkeypatch.setattr(shap, "TreeExplainer", boom)
    monkeypatch.setattr(shap, "LinearExplainer", boom)
    e = ex.ModelExplainer(pre_bundle, background)
    assert e.method == "occlusion"
    assert len(e.explain_row(background.head(1))) == len(pre_bundle.input_features)


def test_explain_row_requires_single_row(pre_bundle, background):
    with pytest.raises(ValueError):
        ex.ModelExplainer(pre_bundle, background).explain_row(background.head(2))


def test_description_states_association_not_causation(pre_bundle, background):
    e = ex.ModelExplainer(pre_bundle, background)
    expl = e.explain_row(background.head(1))
    lines = ex.describe_explanation(expl, probability=0.42, threshold=0.3)
    assert "42.0%" in lines[0] and "above" in lines[0]
    assert lines[-1] == ex.ASSOCIATION_NOTE and "not causes" in lines[-1]
    assert not any("because" in l.lower() or "will subscribe" in l.lower() for l in lines[:-1])


def test_real_shap_additivity_and_scenario_features():
    try:
        pre, post = pr.load_bundle("pre_call"), pr.load_bundle("post_call")
        X, y = dl.split_features_target(dl.load_raw())
    except (pr.ArtifactsMissingError, FileNotFoundError):
        pytest.skip("artifacts/data not available")
    s = dl.make_splits(X, y)
    for b in (pre, post):
        feats = b.input_features
        e = ex.ModelExplainer(b, s.X_train[feats].sample(200, random_state=0))
        assert e.method == "shap_tree"
        rows = s.X_val[feats].head(25)
        sv, base, _ = e.shap_matrix(rows)
        recon = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
        np.testing.assert_allclose(recon, b.base_model.predict_proba(rows)[:, 1], atol=1e-6)
    names_pre = set(ex.ModelExplainer(pre, s.X_train[pre.input_features].head(100)).explain_row(s.X_val[pre.input_features].head(1))["feature"])
    assert "duration" not in names_pre
