"""Shared Streamlit helpers: styling, cached loaders, scenario handling and chart styling."""
import json
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import explain as ex
from src import predict as pr

logger = logging.getLogger(__name__)

PALETTE = {"navy": "#0B2545", "blue": "#2F6FDE", "teal": "#0E8F82", "amber": "#D9921C",
           "red": "#C8452D", "grey": "#8A99AB", "line": "#DCE3EA", "mist": "#F1F5F9"}
SCENARIO_LABELS = {"pre_call": "Pre-call", "post_call": "Post-call"}
SCENARIO_HELP = {
    "pre_call": "Uses only information known before dialling. Call duration is excluded. "
                "This is the scenario for deciding whom to call.",
    "post_call": "Also uses call duration, which is only known after the call ends. "
                 "Useful for understanding conversion, not for choosing whom to call.",
}
PREDICTION_DISCLAIMER = ("This is a probabilistic estimate from a statistical model trained on historical "
                         "campaign data. It is not a guarantee, it does not show that contacting a customer will "
                         "cause a subscription, and it should support, not replace, human judgement.")

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1 {{ font-weight: 650; letter-spacing: -0.01em; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.005em; }}
[data-testid="stMetric"] {{ background: {PALETTE['mist']}; border: 1px solid {PALETTE['line']};
    padding: 14px 16px; border-radius: 8px; }}
[data-testid="stMetricLabel"] p {{ color: #44546A; font-size: 0.85rem; }}
.bs-lede {{ color: #44546A; font-size: 1.05rem; max-width: 70ch; margin-top: -0.4rem; }}
.bs-note {{ border-left: 3px solid {PALETTE['teal']}; background: {PALETTE['mist']}; padding: 10px 14px;
    border-radius: 0 6px 6px 0; margin: 0.4rem 0 1rem 0; max-width: 90ch; }}
.bs-warn {{ border-left: 3px solid {PALETTE['amber']}; background: #FBF4E6; padding: 10px 14px;
    border-radius: 0 6px 6px 0; margin: 0.4rem 0 1rem 0; max-width: 90ch; }}
.bs-brand {{ font-size: 1.25rem; font-weight: 700; color: {PALETTE['navy']}; margin-bottom: 0; }}
.bs-brand span {{ color: {PALETTE['teal']}; }}
.bs-sub {{ color: #5B6B7F; font-size: 0.8rem; margin-top: 0; }}
footer {{ visibility: hidden; }}
</style>
"""


# ------------------------------------------------------------------ page chrome
def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown('<p class="bs-brand">BankSmart <span>AI</span></p>'
                        '<p class="bs-sub">Term-deposit subscription prediction</p>', unsafe_allow_html=True)


def sidebar_scenario():
    st.sidebar.radio("Prediction scenario", list(config.SCENARIOS), key="scenario",
                     format_func=lambda s: SCENARIO_LABELS[s],
                     help="Pre-call: only information known before dialling. Post-call: also uses call duration.")
    st.sidebar.caption(SCENARIO_HELP[current_scenario()])


def current_scenario() -> str:
    s = st.session_state.get("scenario")
    return s if s in config.SCENARIOS else "pre_call"


def page_header(title: str, lede: str = ""):
    st.title(title)
    if lede:
        st.markdown(f'<p class="bs-lede">{lede}</p>', unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="{"bs-warn" if warn else "bs-note"}">{text}</div>', unsafe_allow_html=True)


def scenario_banner(scenario: str):
    if scenario == "pre_call":
        note("<b>Pre-call scenario.</b> Uses only what is known before the call. Call duration is excluded, so "
             "these results reflect what the bank could realistically use to decide whom to contact.")
    else:
        note("<b>Post-call scenario.</b> Includes call duration, which is only known after the call has ended. "
             "Accuracy looks much higher, but this model cannot be used to decide whom to call.", warn=True)


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading model")
def _load_bundle(scenario: str):
    return pr.load_bundle(scenario)


def get_bundle(scenario: str = None):
    scenario = scenario or current_scenario()
    try:
        return _load_bundle(scenario)
    except pr.ArtifactsMissingError as exc:
        st.error("Trained model files were not found for this scenario.")
        st.markdown(str(exc))
        st.code(f"python -m src.train --scenario {scenario}", language="bash")
        st.stop()


@st.cache_data(show_spinner=False)
def read_artifact_csv(scenario: str, name: str, index_col=None):
    path = config.ARTIFACT_DIR / scenario / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_root_csv(name: str):
    path = config.ARTIFACT_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def dataset_summary():
    path = config.ARTIFACT_DIR / "dataset_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def explain_info(scenario: str):
    path = config.ARTIFACT_DIR / scenario / "explain" / "explain_info.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def explain_csv(scenario: str, name: str):
    path = config.ARTIFACT_DIR / scenario / "explain" / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Preparing explainer")
def _explainer(scenario: str):
    bundle = _load_bundle(scenario)
    bg = read_root_csv("background_sample.csv")
    if bg is None:
        return None
    return ex.ModelExplainer(bundle, bg)


def get_explainer(scenario: str = None):
    """Explainer for the scenario's model, or None (with a visible message) if its reference data is missing."""
    scenario = scenario or current_scenario()
    get_bundle(scenario)
    expl = _explainer(scenario)
    if expl is None:
        st.warning("Explanations need artifacts/background_sample.csv. Create it with: python -m src.summary")
    return expl


def available_scenarios() -> list:
    return [s for s in config.SCENARIOS if (config.ARTIFACT_DIR / s / "metadata.json").exists()]


# ------------------------------------------------------------------ formatting / charts
def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=PALETTE["navy"], size=13), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        colorway=[PALETTE["teal"], PALETTE["blue"], PALETTE["amber"], PALETTE["grey"], PALETTE["red"]])
    fig.update_xaxes(gridcolor=PALETTE["line"], zeroline=False, linecolor=PALETTE["line"])
    fig.update_yaxes(gridcolor=PALETTE["line"], zeroline=False, linecolor=PALETTE["line"])
    return fig


def show_fig(fig: go.Figure, **kw):
    st.plotly_chart(style_fig(fig, **kw), width="stretch")


def probability_gauge(prob: float, threshold: float, base_rate: float = None) -> go.Figure:
    steps = [{"range": [0, 100 * threshold], "color": "#E8EEF4"},
             {"range": [100 * threshold, 100], "color": "#CDEBE7"}]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * prob, number={"suffix": "%", "valueformat": ".1f", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": PALETTE["teal"], "thickness": 0.28},
               "steps": steps, "bgcolor": "white", "borderwidth": 0,
               "threshold": {"line": {"color": PALETTE["navy"], "width": 4}, "thickness": 0.9,
                             "value": 100 * threshold}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="white",
                      font=dict(color=PALETTE["navy"]))
    return fig
