import plotly.graph_objects as go
import streamlit as st

from src import campaign as cp
from src import ui

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)
preds = ui.read_artifact_csv(scenario, "test_predictions.csv", index_col=0)

ui.page_header("Campaign Analytics", "How well the model ranks customers for outreach, measured on historical data.")
ui.scenario_banner(scenario)

st.header("Historical evaluation (offline)")
ui.note(cp.OFFLINE_DISCLAIMER, warn=True)
if preds is None:
    st.info("Test predictions not found.")
    st.stop()

y, p = preds["y_true"], preds["proba"]
prev = float(y.mean())
st.markdown(f"Held-out test set: **{len(y):,}** customers, **{int(y.sum()):,}** actual subscribers "
            f"(**{ui.pct(prev)}** base rate). Customers are ranked by predicted probability and the top share is selected.")

tbl = cp.campaign_table(y, p, fractions=(0.10, 0.20, 0.30))
st.dataframe(tbl.assign(outreach_fraction=(100 * tbl.outreach_fraction).round(0).astype(int).astype(str) + "%")[
    ["outreach_fraction", "customers_selected", "subscribers_found", "precision", "recall", "lift",
     "random_expected_subscribers"]].rename(columns={
        "outreach_fraction": "outreach", "customers_selected": "customers selected",
        "subscribers_found": "subscribers found", "random_expected_subscribers": "found by random selection"}).round(3),
    hide_index=True, width="stretch")
st.caption("Lift is precision among selected customers divided by the base rate. "
           "Random selection finds the base rate on average, so its lift is 1.")

share = st.slider("Choose the share of customers to contact", 1, 100, 20, format="%d%%",
                  help="Top customers by predicted probability.") / 100
row = cp.campaign_table(y, p, fractions=(share,)).iloc[0]
c = st.columns(5)
c[0].metric("Customers selected", f"{int(row.customers_selected):,}")
c[1].metric("Subscribers found", f"{int(row.subscribers_found):,}", delta=f"{row.extra_vs_random:+.0f} vs random")
c[2].metric("Precision", ui.pct(row.precision))
c[3].metric("Recall", ui.pct(row.recall))
c[4].metric("Lift", f"{row.lift:.2f}x")

g = cp.gains_curve(y, p)
a, b = st.columns(2)
with a:
    fig = go.Figure()
    fig.add_scatter(x=100 * g.fraction_contacted, y=100 * g.recall, name="model ranking", line_color=ui.PALETTE["teal"])
    fig.add_scatter(x=100 * g.fraction_contacted, y=100 * g.random_recall, name="random selection",
                    line=dict(dash="dash", color=ui.PALETTE["grey"]))
    fig.add_vline(x=100 * share, line_color=ui.PALETTE["navy"])
    fig.update_layout(title="Share of subscribers found", xaxis_title="% of customers contacted", yaxis_title="% of subscribers found")
    ui.show_fig(fig, height=340)
with b:
    fig = go.Figure()
    fig.add_scatter(x=100 * g.fraction_contacted, y=g.lift, name="lift", line_color=ui.PALETTE["blue"])
    fig.add_hline(y=1, line_dash="dash", line_color=ui.PALETTE["grey"], annotation_text="random = 1")
    fig.add_vline(x=100 * share, line_color=ui.PALETTE["navy"])
    fig.update_layout(title="Lift over the base rate", xaxis_title="% of customers contacted", yaxis_title="lift")
    ui.show_fig(fig, height=340, legend=False)

st.divider()
st.header("Rank new customers (future predictions)")
ui.note("This section scores customers whose outcomes are <b>not yet known</b>. The numbers are model estimates, "
        "not results. The historical figures above are the only evidence of how well the ranking works.")
if scenario == "post_call":
    ui.note("The post-call model needs call duration, so it cannot rank customers before they are called. "
            "Switch to the pre-call scenario for outreach planning.", warn=True)
batch = st.session_state.get("batch_results")
if not batch or batch["scenario"] != scenario:
    st.info("Score a file on the Batch Prediction page first (using this scenario), then return here.")
else:
    tab = batch["table"]
    scored = tab[tab["status"] != "invalid"].copy()
    st.caption(f"Using the latest batch: {batch['source']} ({len(scored):,} scored customers).")
    frac = st.slider("Share of this batch to prioritise", 1, 100, 20, format="%d%%", key="future_share") / 100
    top = cp.select_top_fraction(scored, frac)
    top.insert(0, "rank", range(1, len(top) + 1))
    c = st.columns(3)
    c[0].metric("Customers prioritised", f"{len(top):,}")
    c[1].metric("Lowest probability included", ui.pct(float(top["probability"].min())))
    c[2].metric("Expected share subscribing (model estimate)", ui.pct(float(top["probability"].mean())),
                help="Mean predicted probability of the prioritised customers. An estimate, not an observed rate.")
    st.dataframe(top.head(500), hide_index=True, width="stretch")
    st.download_button("Download prioritised list (CSV)", top.to_csv(index=False).encode("utf-8"),
                       "banksmart_priority_list.csv", "text/csv", type="primary")
