import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import data_loader as dl
from src import predict as pr
from src import ui
from src.preprocessing import FeatureMismatchError

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)

ui.page_header("Batch Prediction", "Upload a CSV of customers and download a probability for every valid row.")
ui.scenario_banner(scenario)

with st.expander("Required columns and format", expanded=False):
    st.markdown("The file must be a CSV (comma or semicolon separated) with these exact column names. "
                "Columns are never renamed or guessed, and missing information is never invented.")
    st.code(", ".join(bundle.input_features))
    st.markdown("Extra columns are ignored and reported. Non-numeric or out-of-range numbers make that row invalid "
                "(no prediction). Missing cells and unseen categories are still scored but flagged on the row.")
    template = pd.DataFrame(columns=bundle.input_features)
    st.download_button("Download an empty template (header only)", template.to_csv(index=False).encode("utf-8"),
                       "banksmart_template.csv", "text/csv")

uploaded = st.file_uploader("Customer CSV", type=["csv"], help=f"Up to {config.MAX_UPLOAD_MB} MB and {config.MAX_UPLOAD_ROWS:,} rows.")
demo = ui.read_root_csv("demo_customers.csv")
use_demo = False
if demo is not None:
    use_demo = st.checkbox("Use the demo file instead (200 customers from the public dataset's validation split)",
                           value=False, disabled=uploaded is not None)

df, source_name = None, None
if uploaded is not None:
    try:
        df = dl.read_uploaded_csv(uploaded, uploaded.name)
        source_name = uploaded.name
    except dl.DataValidationError as exc:
        st.error(f"Could not use this file: {exc}")
        st.stop()
elif use_demo:
    df, source_name = demo.drop(columns=["y"], errors="ignore"), "demo_customers.csv"

if df is None:
    st.info("Upload a CSV to begin.")
    st.stop()

threshold = st.slider("Decision threshold", 0.01, 0.99, float(round(bundle.threshold, 3)), 0.005,
                      help="Customers at or above this probability are classed as likely subscribers.")

try:
    results, report = pr.predict_frame(df, bundle, threshold)
except FeatureMismatchError as exc:
    st.error("The file does not have the columns this model needs.")
    st.markdown(str(exc))
    st.stop()
except pr.InputValidationError as exc:
    st.error(str(exc))
    st.stop()

if report.leakage_columns_ignored:
    ui.note(f"The column(s) <b>{', '.join(report.leakage_columns_ignored)}</b> are only known after a call, so the "
            f"pre-call model ignored them.", warn=True)
if "y" in report.ignored_columns:
    st.caption("A column named 'y' was found and ignored. It is not used for prediction.")

st.subheader("Batch summary")
valid = results["status"] != "invalid"
c = st.columns(5)
c[0].metric("Rows", f"{report.n_rows:,}")
c[1].metric("Scored", f"{int(valid.sum()):,}")
c[2].metric("Invalid (not scored)", f"{report.n_invalid:,}")
c[3].metric("Scored with warnings", f"{report.n_with_warnings:,}")
n_pos = int((results["predicted_class"] == 1).sum())
c[4].metric("Flagged as likely", f"{n_pos:,}", help=f"{ui.pct(n_pos / max(int(valid.sum()), 1))} of scored rows")
if report.unseen_categories:
    st.warning("Unseen category values (scored with no information from that field): " +
               "; ".join(f"{k}: {', '.join(v)}" for k, v in report.unseen_categories.items()))
if report.missing_cells:
    st.warning(f"{report.missing_cells:,} missing cell(s). Training medians or a 'missing' level were used for "
               f"scoring, and the affected rows are marked 'warning'.")

if valid.any():
    fig = go.Figure(go.Histogram(x=results.loc[valid, "probability"], nbinsx=40, marker_color=ui.PALETTE["teal"]))
    fig.add_vline(x=threshold, line_color=ui.PALETTE["navy"], line_width=2, annotation_text="threshold")
    fig.update_layout(title="Distribution of predicted probabilities", xaxis_title="probability", yaxis_title="customers")
    ui.show_fig(fig, height=300, legend=False)

out_cols = ["probability", "predicted_class", "predicted_label", "status", "data_quality_note"]
base = df.rename(columns={c: f"{c}_uploaded" for c in df.columns if c in out_cols})
out = base.join(results[out_cols])
st.subheader("Results")
st.dataframe(out.head(1000), width="stretch", hide_index=True)
if len(out) > 1000:
    st.caption(f"Showing the first 1,000 of {len(out):,} rows. The download contains all rows.")
st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                   "banksmart_predictions.csv", "text/csv", type="primary")
st.session_state["batch_results"] = {"scenario": scenario, "table": out, "source": source_name, "threshold": threshold}
ui.note(ui.PREDICTION_DISCLAIMER)
