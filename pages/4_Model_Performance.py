import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

import config
from src import evaluate as ev
from src import ui

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)
meta = bundle.metadata
test_pred = ui.read_artifact_csv(scenario, "test_predictions.csv", index_col=0)
val_pred = ui.read_artifact_csv(scenario, "val_predictions.csv", index_col=0)

ui.page_header("Model Performance", "How well the selected model separates subscribers from non-subscribers, and what it costs.")
ui.scenario_banner(scenario)

t = meta["metrics_test"]
tab_cmp, tab_cm, tab_curves, tab_thr, tab_cal, tab_cv = st.tabs(
    ["Model comparison", "Confusion matrix and report", "ROC and PR curves", "Threshold tuning", "Calibration",
     "Cross-validation"])

# ---------------------------------------------------------------- comparison
with tab_cmp:
    comp = ui.read_artifact_csv(scenario, "model_comparison.csv")
    st.markdown("All models are scored on the **same** validation and test splits. Each model uses its own "
                f"threshold, chosen on validation data to maximise F{config.THRESHOLD_BETA:g}. "
                "The selected model is the one with the highest **validation** PR-AUC.")
    cols = ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "threshold", "n_predicted_positive"]
    for name, sp in (("Validation (used for selection)", "val"), ("Test (reporting only)", "test")):
        st.markdown(f"**{name}**")
        d = comp[comp.split == sp][cols].copy().sort_values("pr_auc", ascending=False)
        d.insert(0, "selected", np.where(d["model"] == meta["selected_model"], "yes", ""))
        d["model"] = d["model"].str.replace("_", " ")
        st.dataframe(d.round(4), hide_index=True, width="stretch")
    st.markdown("**Why accuracy alone is not enough.** Only "
                f"{ui.pct(t['prevalence'])} of customers subscribe, so a model that predicts 'no' for everyone is "
                f"{ui.pct(1 - t['prevalence'])} accurate while finding no subscribers at all. "
                f"The selected model's accuracy is {ui.pct(t['accuracy'])}, but its recall "
                f"({ui.pct(t['recall'])}) and precision ({ui.pct(t['precision'])}) say far more about its usefulness.")

# ---------------------------------------------------------------- confusion + report
with tab_cm:
    left, right = st.columns([1, 1])
    with left:
        z = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
        txt = [[f"{v:,}" for v in r] for r in z]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted: no", "Predicted: yes"], y=["Actual: no", "Actual: yes"],
                                   text=txt, texttemplate="%{text}", colorscale=[[0, "#F1F5F9"], [1, ui.PALETTE["teal"]]],
                                   showscale=False, textfont={"size": 18}))
        fig.update_layout(title=f"Test set at threshold {t['threshold']:.3f}")
        fig.update_yaxes(autorange="reversed")
        ui.show_fig(fig, height=340, legend=False)
    with right:
        st.markdown("**Trade-off between the two kinds of error**")
        st.markdown(f"- **False positives ({t['fp']:,})**: customers flagged who did not subscribe. Each is a wasted "
                    f"call, and too many lower precision.\n"
                    f"- **False negatives ({t['fn']:,})**: subscribers the model missed. Each is a lost opportunity, "
                    f"and too many lower recall.\n"
                    f"- Raising the threshold removes false positives but adds false negatives; lowering it does the "
                    f"reverse. The default favours recall (F{config.THRESHOLD_BETA:g}), assuming a missed subscriber "
                    f"costs more than an extra call. No cost data was available to confirm that assumption.")
    rep = meta["classification_report_test"]
    rows = []
    for k in ("not_subscribed", "subscribed"):
        rows.append({"class": k.replace("_", " "), "precision": rep[k]["precision"], "recall": rep[k]["recall"],
                     "f1-score": rep[k]["f1-score"], "support": int(rep[k]["support"])})
    st.markdown("**Classification report (test set)**")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    ci = meta["test_bootstrap_ci_95"]
    st.markdown("**Test metrics with 95% bootstrap confidence intervals**")
    st.dataframe(pd.DataFrame([{"metric": k, "value": v["point"], "ci low": v["low"], "ci high": v["high"]}
                               for k, v in ci.items()]).round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- curves
with tab_curves:
    if test_pred is None:
        st.info("Test predictions not found.")
    else:
        y, p = test_pred["y_true"].to_numpy(), test_pred["proba"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"model (AUC {t['roc_auc']:.3f})", line_color=ui.PALETTE["teal"])
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dash", color=ui.PALETTE["grey"]))
            fig.update_layout(title="ROC curve", xaxis_title="false positive rate", yaxis_title="true positive rate")
            ui.show_fig(fig, height=360)
        with b:
            fig = go.Figure()
            fig.add_scatter(x=rec, y=prec, mode="lines", name=f"model (PR-AUC {t['pr_auc']:.3f})", line_color=ui.PALETTE["teal"])
            fig.add_hline(y=t["prevalence"], line_dash="dash", line_color=ui.PALETTE["grey"],
                          annotation_text=f"prevalence {t['prevalence']:.3f}", annotation_position="top right")
            fig.update_layout(title="Precision-recall curve", xaxis_title="recall", yaxis_title="precision")
            ui.show_fig(fig, height=360)
        st.caption("With rare positives the precision-recall curve is the more informative view: the dashed line is "
                   "what random guessing would achieve.")

# ---------------------------------------------------------------- threshold tuning
with tab_thr:
    split = st.radio("Data", ["validation", "test"], horizontal=True,
                     help="The threshold is chosen on validation data. The test view shows how that choice held up.")
    sweep = ui.read_artifact_csv(scenario, "threshold_sweep_val.csv" if split == "validation" else "threshold_sweep_test.csv")
    pred = val_pred if split == "validation" else test_pred
    if sweep is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for col, color in (("precision", ui.PALETTE["blue"]), ("recall", ui.PALETTE["teal"]), ("f1", ui.PALETTE["amber"])):
            fig.add_scatter(x=sweep["threshold"], y=sweep[col], name=col, mode="lines", line_color=color, secondary_y=False)
        fig.add_scatter(x=sweep["threshold"], y=sweep["n_predicted_positive"], name="customers flagged", mode="lines",
                        line=dict(color=ui.PALETTE["grey"], dash="dot"), secondary_y=True)
        fig.add_vline(x=meta["threshold"], line_color=ui.PALETTE["navy"], annotation_text="chosen threshold")
        fig.update_yaxes(title_text="score", secondary_y=False)
        fig.update_yaxes(title_text="customers flagged", secondary_y=True, showgrid=False)
        fig.update_layout(xaxis_title="decision threshold")
        ui.show_fig(fig, height=400)
        thr = st.slider("Inspect a threshold", 0.02, 0.98, float(round(meta["threshold"], 2)), 0.01)
        if pred is not None:
            m = ev.full_metrics(pred["y_true"], pred["proba"], thr)
            c = st.columns(5)
            c[0].metric("Precision", f"{m['precision']:.3f}")
            c[1].metric("Recall", f"{m['recall']:.3f}")
            c[2].metric("F1", f"{m['f1']:.3f}")
            c[3].metric("Customers flagged", f"{m['n_predicted_positive']:,}")
            c[4].metric("Missed subscribers", f"{m['fn']:,}")

# ---------------------------------------------------------------- calibration
with tab_cal:
    final = ui.read_artifact_csv(scenario, "reliability_final_test.csv")
    raw = ui.read_artifact_csv(scenario, "reliability_uncalibrated_test.csv")
    cal = ui.read_artifact_csv(scenario, "calibration_comparison.csv")
    if final is not None:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated", line=dict(dash="dash", color=ui.PALETTE["grey"]))
        if raw is not None:
            fig.add_scatter(x=raw["mean_predicted"], y=raw["fraction_positive"], mode="lines+markers",
                            name="before calibration", line_color=ui.PALETTE["blue"])
        fig.add_scatter(x=final["mean_predicted"], y=final["fraction_positive"], mode="lines+markers",
                        name=f"final model ({meta['calibration_method']})", line_color=ui.PALETTE["teal"])
        fig.update_layout(title="Reliability on the test set", xaxis_title="mean predicted probability",
                          yaxis_title="observed subscription rate")
        ui.show_fig(fig, height=380)
        st.markdown(f"Test-set Brier score **{t['brier']:.4f}**, expected calibration error **{t['ece']:.4f}**. "
                    "Closeness to the diagonal means a predicted 20% really corresponds to about 20% of such customers "
                    "subscribing.")
    if cal is not None:
        st.markdown("**Calibration methods compared on validation data** (lowest Brier score is chosen)")
        st.dataframe(cal.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- cross-validation
with tab_cv:
    cv = ui.read_artifact_csv(scenario, "cv_results.csv")
    if cv is not None:
        st.markdown(f"{meta['training_config']['cv_folds']}-fold stratified cross-validation on the training split only. "
                    "Recall and precision here use the default 0.5 cut-off, which is why class-weighted models look "
                    "so different from unweighted ones; PR-AUC and ROC-AUC do not depend on a threshold.")
        d = cv.copy()
        fig = go.Figure(go.Bar(x=d["model"].str.replace("_", " "), y=d["pr_auc_mean"],
                               error_y=dict(type="data", array=d["pr_auc_std"]), marker_color=ui.PALETTE["teal"]))
        fig.update_layout(title="Cross-validated PR-AUC (mean, error bars = 1 standard deviation)", xaxis_tickangle=-25)
        ui.show_fig(fig, height=360, legend=False)
        show = pd.DataFrame({"model": d["model"].str.replace("_", " ")})
        for m in ("pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy"):
            show[m] = [f"{a:.3f} ± {s:.3f}" for a, s in zip(d[f"{m}_mean"], d[f"{m}_std"])]
        st.dataframe(show, hide_index=True, width="stretch")
