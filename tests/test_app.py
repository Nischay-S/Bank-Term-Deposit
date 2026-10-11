"""Functional tests of the Streamlit pages using Streamlit's headless AppTest harness.
(File uploads are not supported by AppTest; the upload parser is unit-tested in test_upload.py.)"""
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from src import predict as pr

PAGES = ["1_Executive_Dashboard", "2_Customer_Prediction", "3_Batch_Prediction", "4_Model_Performance",
         "5_Explainable_AI", "6_Campaign_Analytics", "7_About_Project"]
ROOT = config.ROOT

pytestmark = pytest.mark.skipif(
    not all((config.ARTIFACT_DIR / s / "metadata.json").exists() for s in config.SCENARIOS),
    reason="trained artifacts not present")


def _app(page, scenario="pre_call"):
    at = AppTest.from_file(str(ROOT / "pages" / f"{page}.py"), default_timeout=120)
    at.session_state["scenario"] = scenario
    return at


@pytest.mark.parametrize("scenario", config.SCENARIOS)
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, scenario):
    at = _app(page, scenario).run()
    assert not at.exception, [e.value for e in at.exception]


def test_main_app_renders():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception


@pytest.mark.parametrize("scenario", config.SCENARIOS)
def test_single_prediction_flow(scenario):
    at = _app("2_Customer_Prediction", scenario).run()
    labels_before = [n.label for n in at.number_input]
    assert ("Call duration (seconds)" in labels_before) == (scenario == "post_call")   # duration only post-call
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = {m.label: m.value for m in at.metric}
    assert "Estimated probability of subscribing" in metrics
    prob = float(metrics["Estimated probability of subscribing"].rstrip("%")) / 100
    assert 0 <= prob <= 1
    assert any("probabilistic estimate" in str(m.value) for m in at.markdown)   # disclaimer shown


def test_batch_demo_flow_stores_results():
    at = _app("3_Batch_Prediction").run()
    at.checkbox[0].check().run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "200" and metrics["Invalid (not scored)"] == "0"
    stored = at.session_state["batch_results"]
    assert stored["scenario"] == "pre_call" and len(stored["table"]) == 200
    assert stored["table"]["probability"].between(0, 1).all()


def test_campaign_page_ranks_a_batch():
    bundle = pr.load_bundle("pre_call")
    demo = pd.read_csv(config.ARTIFACT_DIR / "demo_customers.csv").drop(columns=["y"])
    res, _ = pr.predict_frame(demo, bundle)
    at = _app("6_Campaign_Analytics")
    at.session_state["batch_results"] = {"scenario": "pre_call", "table": demo.join(res), "source": "t.csv",
                                         "threshold": bundle.threshold}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Customers prioritised" in labels and "Lift" in labels


def test_campaign_page_ignores_batch_from_other_scenario():
    at = _app("6_Campaign_Analytics")
    at.session_state["batch_results"] = {"scenario": "post_call", "table": pd.DataFrame(), "source": "x", "threshold": 0.1}
    at.run()
    assert not at.exception
    assert "Customers prioritised" not in [m.label for m in at.metric]


def test_missing_artifacts_show_message_not_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    st.cache_resource.clear(); st.cache_data.clear()
    at = _app("2_Customer_Prediction").run()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)
    assert any("python -m src.train" in c.value for c in at.code)
    st.cache_resource.clear(); st.cache_data.clear()
