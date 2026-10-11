import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import explain as ex
from src import predict as pr
from src import ui

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)
info = ui.explain_info(scenario)

ui.page_header("Explainable AI", "Which factors the model relies on, for the whole customer base and for one customer.")
ui.scenario_banner(scenario)
ui.note("<b>Read these as model behaviour, not causes.</b> An explanation says how the model used an input to reach its "
        "estimate, based on patterns in historical data. It does not show that changing a customer's attribute would "
        "change their decision.")
if info:
    st.caption(f"Method: **{info['method'].replace('_', ' ')}** on a {info['estimator']} "
               f"({info['n_explained']:,} validation customers). Contributions are in {info['output_space']} units.")

tab_g, tab_c, tab_i = st.tabs(["Global importance", "Cross-checks", "Individual customer"])

with tab_g:
    g = ui.explain_csv(scenario, "global_shap.csv")
    if g is None:
        st.info("Global SHAP importance not found. Create it with: python -m src.explain")
    else:
        top = g.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["label"], orientation="h", marker_color=ui.PALETTE["teal"]))
        fig.update_layout(title="Average impact on the model's score (mean absolute SHAP)", xaxis_title="mean |SHAP| (log-odds)")
        ui.show_fig(fig, height=420, legend=False)
        st.caption("One-hot encoded categories are added back together, so each bar is one original feature.")
    npz = config.ARTIFACT_DIR / scenario / "explain" / "shap_sample.npz"
    if npz.exists():
        st.markdown("**SHAP summary plot.** Each dot is one customer. Position shows the effect on the score; "
                    "colour shows whether that input value was high or low.")
        try:
            import matplotlib.pyplot as plt
            fig = ex.shap_summary_figure(npz)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:  # documented fallback instead of crashing
            st.warning(f"The SHAP summary plot could not be drawn ({type(exc).__name__}). The bar chart above shows the same ranking.")

with tab_c:
    st.markdown("Two independent checks on the ranking above.")
    p = ui.explain_csv(scenario, "permutation_importance.csv")
    if p is not None:
        top = p.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["pr_auc_drop_mean"], y=top["feature"].map(ex.label), orientation="h",
                               error_x=dict(type="data", array=top["pr_auc_drop_std"]), marker_color=ui.PALETTE["blue"]))
        fig.update_layout(title="Permutation importance: PR-AUC lost when a feature is shuffled",
                          xaxis_title="drop in PR-AUC (validation)")
        ui.show_fig(fig, height=420, legend=False)
        st.caption("This works for any model, so it still applies if SHAP is unavailable. It can rank correlated "
                   "features differently from SHAP.")
    lr = ui.explain_csv(scenario, "logreg_coefficients.csv")
    if lr is not None:
        st.markdown("**Logistic regression reference** (an interpretable model trained on the same data, not the model "
                    "used for predictions). Numeric inputs are standardised; a category's coefficient is relative to "
                    "the other levels of that feature.")
        a, b = st.columns(2)
        a.markdown("Largest positive coefficients")
        a.dataframe(lr.sort_values("coefficient", ascending=False).head(8).round(3), hide_index=True, width="stretch")
        b.markdown("Largest negative coefficients")
        b.dataframe(lr.sort_values("coefficient").head(8).round(3), hide_index=True, width="stretch")

with tab_i:
    demo = ui.read_root_csv("demo_customers.csv")
    explainer = ui.get_explainer(scenario)
    if demo is None or explainer is None:
        st.info("Demo customers not found. Create them with: python -m src.summary")
    else:
        n = len(demo)
        idx = st.number_input("Demo customer (from the validation split)", 1, n, 1) - 1
        row = demo.iloc[[idx]]
        st.dataframe(row.drop(columns=["y"]), hide_index=True, width="stretch")
        res = pr.predict_single(row.drop(columns=["y"]).iloc[0].to_dict(), bundle)
        expl = explainer.explain_row(row[bundle.input_features])
        c = st.columns(3)
        c[0].metric("Estimated probability", ui.pct(res["probability"]))
        c[1].metric("Predicted class", "Likely subscriber" if res["predicted_class"] else "Not flagged")
        c[2].metric("Historical outcome", "Subscribed" if row.iloc[0]["y"] == "yes" else "Did not subscribe",
                    help="From the dataset. Shown for context only; the model did not see it.")
        pos = expl[expl.contribution > 0].head(5)
        neg = expl[expl.contribution < 0].head(5)
        a, b = st.columns(2)
        for col, d, title, colr in ((a, pos, "Strongest upward contributions", ui.PALETTE["teal"]),
                                    (b, neg, "Strongest downward contributions", ui.PALETTE["amber"])):
            with col:
                st.markdown(f"**{title}**")
                if d.empty:
                    st.caption("None.")
                else:
                    t = d.iloc[::-1]
                    fig = go.Figure(go.Bar(x=t["contribution"], y=[f"{ex.label(f)} = {v}" for f, v in zip(t["feature"], t["value"])],
                                           orientation="h", marker_color=colr))
                    ui.show_fig(fig, height=250, legend=False)
        for line in ex.describe_explanation(expl, res["probability"], bundle.threshold):
            st.markdown(f"- {line}")
