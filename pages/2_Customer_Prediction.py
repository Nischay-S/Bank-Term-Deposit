import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import explain as ex
from src import predict as pr
from src import ui

scenario = ui.current_scenario()
bundle = ui.get_bundle(scenario)
summary = ui.dataset_summary()
base_rate = summary["prevalence"] if summary else bundle.metadata["metrics_test"]["prevalence"]

ui.page_header("Customer Prediction", "Enter one customer's details to estimate the probability of subscribing.")
ui.scenario_banner(scenario)

cats = {c: sorted(bundle.known_categories.get(c) or config.CATEGORY_DOMAINS.get(c, [])) for c in config.CATEGORICAL_FEATURES}
defaults = {"job": "management", "marital": "married", "education": "secondary", "default": "no",
            "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "poutcome": "unknown"}
tips = {
    "balance": "Average yearly account balance in euros. Can be negative.",
    "campaign": "Number of contacts with this customer in the current campaign, including this one.",
    "pdays": "Days since the customer was last contacted in a previous campaign.",
    "previous": "Contacts with this customer before the current campaign.",
    "duration": "Length of the call in seconds. Only known after the call ends.",
    "contact": "How the customer was reached. 'unknown' means the channel was not recorded.",
}


def pick(label, col):
    opts = cats[col]
    idx = opts.index(defaults[col]) if defaults[col] in opts else 0
    return st.selectbox(label, opts, index=idx, help=tips.get(col))


with st.form("customer_form"):
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Customer profile**")
        age = st.number_input("Age", 18, 100, 40)
        job = pick("Occupation", "job")
        marital = pick("Marital status", "marital")
        education = pick("Education", "education")
        credit_default = pick("Credit in default", "default")
        balance = st.number_input("Account balance (EUR)", -100_000, 1_000_000, 500, step=100, help=tips["balance"])
        housing = pick("Housing loan", "housing")
        loan = pick("Personal loan", "loan")
    with b:
        st.markdown("**Current contact**")
        contact = pick("Contact channel", "contact")
        day = st.number_input("Day of month", 1, 31, 15)
        month = pick("Month", "month")
        campaign = st.number_input("Contacts in this campaign", 1, 100, 1, help=tips["campaign"])
        duration = None
        if scenario == "post_call":
            duration = st.number_input("Call duration (seconds)", 0, 10_000, 180, help=tips["duration"])
    with c:
        st.markdown("**Contact history**")
        was_contacted = st.checkbox("Contacted in a previous campaign", value=False)
        pdays_in = st.number_input("Days since previous contact", 1, 1000, 90, help=tips["pdays"] +
                                   " Ignored unless 'contacted in a previous campaign' is ticked.")
        previous = st.number_input("Earlier contacts", 0, 100, 0, help=tips["previous"])
        poutcome = pick("Outcome of previous campaign", "poutcome")
    submitted = st.form_submit_button("Estimate probability", type="primary")

if submitted:
    values = {"age": age, "job": job, "marital": marital, "education": education, "default": credit_default,
              "balance": balance, "housing": housing, "loan": loan, "contact": contact, "day": day,
              "month": month, "campaign": campaign, "pdays": pdays_in if was_contacted else -1,
              "previous": previous, "poutcome": poutcome}
    if duration is not None:
        values["duration"] = duration
    st.session_state["last_customer"] = (scenario, values)

threshold = st.slider("Decision threshold", 0.01, 0.99, float(round(bundle.threshold, 3)), 0.005,
                      help="Customers at or above this probability are classed as likely subscribers. The default was "
                           "chosen on validation data to favour recall (assumption: a missed subscriber costs more "
                           "than an extra call).")

stored = st.session_state.get("last_customer")
if not stored or stored[0] != scenario:
    st.info("Fill in the form and press **Estimate probability**.")
    st.stop()
values = stored[1]

if not values.get("previous", 0) == 0 and values["pdays"] == -1:
    ui.note("You entered earlier contacts but did not tick 'Contacted in a previous campaign'. "
            "In the data these two almost always agree; check the history fields.", warn=True)

try:
    result = pr.predict_single(values, bundle, threshold)
except pr.InputValidationError as exc:
    st.error(f"Input problem: {exc}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.plotly_chart(ui.probability_gauge(result["probability"], threshold), width="stretch")
    st.caption("Dark marker: decision threshold. Teal zone: classed as likely subscriber.")
with right:
    st.metric("Estimated probability of subscribing", ui.pct(result["probability"]))
    verdict = "Likely subscriber" if result["predicted_class"] == 1 else "Not flagged"
    st.metric("Predicted class", verdict)
    ratio = result["probability"] / base_rate
    st.markdown(f"The historical subscription rate is **{ui.pct(base_rate)}**, so this customer's estimate is "
                f"**{ratio:.1f} times** the average.")
if result["notes"]:
    ui.note(result["notes"], warn=True)

# ---- explanation ----
st.subheader("Why the model gave this estimate")
explainer = ui.get_explainer(scenario)
if explainer is not None:
    row = pd.DataFrame([values])[bundle.input_features]
    expl = explainer.explain_row(row)
    top = expl.head(10).iloc[::-1]
    labels = [f"{ex.label(f)} = {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{ex.label(f)} = {v}"
              for f, v in zip(top["feature"], top["value"])]
    fig = go.Figure(go.Bar(x=top["contribution"], y=labels, orientation="h",
                           marker_color=[ui.PALETTE["teal"] if v > 0 else ui.PALETTE["amber"] for v in top["contribution"]]))
    unit = "log-odds" if expl.attrs["output_space"] == "log-odds" else "probability points"
    fig.update_layout(title="Strongest contributions", xaxis_title=f"Effect on the model's score ({unit})")
    ui.show_fig(fig, height=380, legend=False)
    st.caption("Teal bars raised the estimate; amber bars lowered it. Contributions are for the underlying model "
               "before calibration, and are measured against an average customer.")
    for line in ex.describe_explanation(expl, result["probability"], threshold):
        st.markdown(f"- {line}")

ui.note(ui.PREDICTION_DISCLAIMER)
