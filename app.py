"""BankSmart AI - Streamlit entry point.  Run with:  streamlit run app.py"""
import streamlit as st

import config
from src import ui

st.set_page_config(page_title="BankSmart AI", page_icon=None, layout="wide", initial_sidebar_state="expanded")
ui.inject_css()


def overview():
    ui.page_header("BankSmart AI",
                   "Estimate which bank customers are most likely to subscribe to a term deposit when "
                   "contacted by phone, and see why the model thinks so.")
    summary = ui.dataset_summary()
    c1, c2, c3 = st.columns(3)
    if summary:
        c1.metric("Historical contacts", f"{summary['n_rows']:,}")
        c2.metric("Subscribed", ui.pct(summary["prevalence"]), help="Share of past contacts that ended in a subscription.")
    scenarios = ui.available_scenarios()
    c3.metric("Trained scenarios", f"{len(scenarios)} of {len(config.SCENARIOS)}")
    if not scenarios:
        st.error("No trained models found. Create them with the commands below, then reload this page.")
        st.code("python -m src.train --scenario all\npython -m src.explain --scenario all\n"
                "python -m src.campaign\npython -m src.summary", language="bash")

    st.subheader("Two scenarios, chosen in the sidebar")
    st.markdown(
        "**Pre-call** uses only what the bank knows before dialling, so it is the scenario for deciding whom to "
        "contact. **Post-call** also uses the length of the call. That information is only available after the "
        "call, so the post-call model is shown for comparison and analysis, not for choosing whom to call.")

    st.subheader("What each page does")
    st.markdown(
        "- **Executive Dashboard**: dataset facts, headline metrics and model comparison.\n"
        "- **Customer Prediction**: enter one customer and get a probability with an explanation.\n"
        "- **Batch Prediction**: upload a CSV and download predictions for every valid row.\n"
        "- **Model Performance**: confusion matrix, ROC and precision-recall curves, calibration, threshold tuning.\n"
        "- **Explainable AI**: which factors drive the model, globally and per customer.\n"
        "- **Campaign Analytics**: how well the model ranks customers on historical data, and how to rank new ones.\n"
        "- **About the Project**: problem, method, limitations and ethics.")

    st.subheader("Method in brief")
    st.markdown(
        "Rows are split at random into training, validation and test sets (60/20/20, stratified, fixed seed). "
        "Candidate models are compared by cross-validation on the training set. The model, probability calibration "
        "and decision threshold are chosen on validation data only. The test set is scored after those choices, "
        "and results carry bootstrap confidence intervals.")

    st.subheader("Limitations to keep in mind")
    ui.note("All results are offline measurements on one historical dataset from a Portuguese bank (2008 to 2010). "
            "They show how well the model ranks past customers, not what would happen in a new campaign. "
            "The data is ordered by date but a random split was used, so a later period may be harder to predict. "
            "No revenue, ROI or uplift claim is made, because no cost or experiment data exists. "
            "See <b>About the Project</b> for the full list.")


pages = [
    st.Page(overview, title="Overview", default=True, url_path="overview"),
    st.Page("pages/1_Executive_Dashboard.py", title="Executive Dashboard", url_path="dashboard"),
    st.Page("pages/2_Customer_Prediction.py", title="Customer Prediction", url_path="predict"),
    st.Page("pages/3_Batch_Prediction.py", title="Batch Prediction", url_path="batch"),
    st.Page("pages/4_Model_Performance.py", title="Model Performance", url_path="performance"),
    st.Page("pages/5_Explainable_AI.py", title="Explainable AI", url_path="explain"),
    st.Page("pages/6_Campaign_Analytics.py", title="Campaign Analytics", url_path="campaign"),
    st.Page("pages/7_About_Project.py", title="About the Project", url_path="about"),
]
navigation = st.navigation(pages)
ui.sidebar_brand()
ui.sidebar_scenario()
navigation.run()
