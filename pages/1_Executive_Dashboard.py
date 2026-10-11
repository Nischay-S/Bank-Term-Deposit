import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import ui

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)
meta = bundle.metadata
summary = ui.dataset_summary()

ui.page_header("Executive Dashboard", "Dataset, headline results and model comparison for the selected scenario.")
ui.scenario_banner(scenario)

# ---- headline metrics (held-out test set) ----
st.subheader("Selected model on the held-out test set")
t, ci = meta["metrics_test"], meta["test_bootstrap_ci_95"]
st.caption(f"Model: **{meta['selected_model'].replace('_', ' ')}**, calibration: **{meta['calibration_method']}**, "
           f"decision threshold **{meta['threshold']:.3f}** ({meta['threshold_objective']}). "
           f"Chosen on validation data; the test set was scored afterwards.")
cols = st.columns(5)
for col, (key, name, tip) in zip(cols, [
        ("pr_auc", "PR-AUC", "Area under the precision-recall curve. The key ranking metric when subscribers are rare."),
        ("roc_auc", "ROC-AUC", "Chance that a random subscriber is scored above a random non-subscriber."),
        ("recall", "Recall", "Share of actual subscribers the model flags at the chosen threshold."),
        ("precision", "Precision", "Share of flagged customers who actually subscribed."),
        ("f1", "F1-score", "Harmonic mean of precision and recall.")]):
    col.metric(name, f"{t[key]:.3f}", help=tip)
    col.caption(f"95% CI {ci[key]['low']:.3f} to {ci[key]['high']:.3f}")

# ---- dataset ----
st.subheader("Dataset")
if summary:
    c = st.columns(4)
    c[0].metric("Customers", f"{summary['n_rows']:,}")
    c[1].metric("Input features", len(bundle.input_features),
                help="Raw inputs used by this scenario. Post-call adds call duration.")
    c[2].metric("Subscribers", f"{summary['n_positive']:,}")
    c[3].metric("Subscription rate", ui.pct(summary["prevalence"]))
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Bar(x=["Did not subscribe", "Subscribed"],
                               y=[summary["n_rows"] - summary["n_positive"], summary["n_positive"]],
                               marker_color=[ui.PALETTE["blue"], ui.PALETTE["teal"]],
                               text=[f"{summary['n_rows'] - summary['n_positive']:,}", f"{summary['n_positive']:,}"],
                               textposition="outside"))
        fig.update_layout(title="Class distribution")
        ui.show_fig(fig, height=340, legend=False)
    with right:
        feature = st.selectbox("Subscription rate by", ["month", "poutcome", "contact", "job", "education",
                                                         "marital", "housing", "loan", "default"],
                               help="Share of contacts that ended in a subscription, per category.")
        d = pd.DataFrame(summary["rates_by_category"][feature])
        fig = go.Figure(go.Bar(x=d["category"], y=100 * d["rate"], marker_color=ui.PALETTE["teal"],
                               customdata=d["n"], hovertemplate="%{x}: %{y:.1f}% of %{customdata:,} contacts<extra></extra>"))
        fig.add_hline(y=100 * summary["prevalence"], line_dash="dash", line_color=ui.PALETTE["navy"],
                      annotation_text="overall rate", annotation_position="top left")
        fig.update_layout(title=f"Subscription rate by {feature}", yaxis_title="% subscribed")
        ui.show_fig(fig, height=340, legend=False)
    u = summary["unknown_share"]
    ui.note(f"<b>Data notes.</b> {ui.pct(summary['never_previously_contacted_share'])} of customers were never contacted "
            f"in an earlier campaign, so <code>poutcome</code> is 'unknown' for {ui.pct(u['poutcome'])} and the "
            f"contact channel is 'unknown' for {ui.pct(u['contact'])}. Month shows strong swings in subscription "
            f"rate, which may reflect how campaigns were run rather than customer traits.")
else:
    st.info("Dataset summary not found. Create it with: python -m src.summary")

# ---- model comparison ----
st.subheader("Model comparison")
comp = ui.read_artifact_csv(scenario, "model_comparison.csv")
if comp is not None:
    split = st.radio("Evaluated on", ["validation", "test"], horizontal=True,
                     help="Models are selected on validation data. Test results are shown for reporting only.")
    d = comp[comp.split == ("val" if split == "validation" else "test")].copy()
    d["model"] = d["model"].str.replace("_", " ")
    fig = go.Figure()
    fig.add_bar(name="PR-AUC", x=d["model"], y=d["pr_auc"], marker_color=ui.PALETTE["teal"])
    fig.add_bar(name="ROC-AUC", x=d["model"], y=d["roc_auc"], marker_color=ui.PALETTE["blue"])
    fig.update_layout(barmode="group", yaxis_title="score", xaxis_tickangle=-25)
    ui.show_fig(fig, height=380)
    sel = meta["selected_model"].replace("_", " ")
    st.caption(f"Selected by highest validation PR-AUC: **{sel}**. Differences between the top models are small "
               f"compared with cross-validation spread, so treat the ranking among them as tentative.")
else:
    st.info("Model comparison table not found.")

# ---- pre vs post ----
both = ui.available_scenarios()
if len(both) == 2:
    st.subheader("Pre-call versus post-call")
    rows = []
    for s in both:
        m = ui.get_bundle(s).metadata["metrics_test"]
        rows.append({"Scenario": ui.SCENARIO_LABELS[s], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
                     "Recall": m["recall"], "Precision": m["precision"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("The post-call gain comes mainly from call duration, which is not known when deciding whom to call.")
