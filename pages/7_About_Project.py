import streamlit as st

import config
from src import ui

scenario = ui.current_scenario()
summary = ui.dataset_summary()
meta = ui.get_bundle(scenario).metadata

ui.page_header("About the Project", "Problem, method, limitations and ethics.")
tabs = st.tabs(["Problem and objectives", "Data and method", "Limitations", "Ethics", "Future scope", "Technology"])

with tabs[0]:
    st.markdown("""
**Problem.** Banks contact customers by phone to offer term deposits. Most calls do not result in a subscription, so
calls are costly and many are wasted. The question is whether information the bank already holds can tell us which
customers are more likely to subscribe.

**Objectives**
- Build and compare classification models for term-deposit subscription, treating the rare positive class carefully.
- Separate a **pre-call** scenario (usable for deciding whom to call) from a **post-call** scenario that also uses call duration.
- Choose the model, calibration and decision threshold on validation data, and report honest test results with confidence intervals.
- Explain predictions for the whole customer base and for individual customers.
- Provide a usable application with single, batch and campaign-ranking views.
""")

with tabs[1]:
    st.markdown("""
**Dataset.** UCI Machine Learning Repository, *Bank Marketing* (dataset 222), direct phone marketing campaigns of a
Portuguese bank, released under CC BY 4.0. Original study: S. Moro, P. Cortez and P. Rita, "A Data-Driven Approach to
Predict the Success of Bank Telemarketing", *Decision Support Systems*, 62, 22-31, 2014.
""")
    if summary:
        st.markdown(f"This project uses the 17-column version: **{summary['n_rows']:,}** contacts, "
                    f"**{ui.pct(summary['prevalence'])}** subscribed. It has no economic indicators.")
    tc = meta["training_config"]
    st.markdown(f"""
**Method**
1. Stratified random split: train {tc['n_train']:,}, validation {tc['n_val']:,}, test {tc['n_test']:,} (seed {tc['seed']}).
2. Preprocessing inside the model pipeline (imputation, scaling, one-hot encoding), so nothing leaks from validation or test data.
3. Models: logistic regression (plain, class-weighted, SMOTE inside the pipeline), decision tree, random forest, gradient boosting, plus lightly tuned forest and boosting.
4. {tc['cv_folds']}-fold cross-validation on the training split; selection by validation PR-AUC.
5. Probability calibration (none, sigmoid, isotonic) chosen by validation Brier score.
6. Threshold chosen on validation data to maximise F{config.THRESHOLD_BETA:g}, which favours recall.
7. One final evaluation on the test split with bootstrap confidence intervals.
8. SHAP, permutation importance and a logistic reference for explanation; lift analysis for campaign ranking.
""")

with tabs[2]:
    st.markdown("""
- **Offline only.** Results measure ranking on historical data. They do not show that contacting top-ranked customers causes additional subscriptions. Revenue, ROI and uplift cannot be claimed without cost data and a controlled experiment.
- **One bank, one period.** The data covers a Portuguese bank from 2008 to 2010. Customer behaviour and market conditions have changed since.
- **Random split, but the data is time-ordered.** The main results use a stratified random split, which treats rows as independent. The dataset documentation says the file is ordered by date, and the month sequence in the file agrees. A random split lets the model learn from later periods, so the headline numbers are likely optimistic for forecasting a future campaign. The model also relies heavily on month and day, which may mark campaign waves. A supplementary time-ordered check (below) shows noticeably lower ranking performance.
- **Missing context.** There are no economic indicators and no cost or revenue data. The recall-favouring threshold is an assumption.
- **Data quality.** `poutcome` is unknown for most customers and the contact channel is unknown for many; "unknown" is treated as its own category.
- **Model comparison is tentative.** The top models differ by less than the cross-validation spread, and tuning was deliberately light.
- **SMOTE** was applied to one-hot encoded columns, so synthetic rows contain fractional category values.
- **Explanations** describe model behaviour, not causal effects.
""")

with tabs[2]:
    import json as _json
    _p = config.ARTIFACT_DIR / "pre_call" / "temporal_check.json"
    if _p.exists():
        _t = _json.loads(_p.read_text())
        st.markdown("**Supplementary time-ordered check (pre-call, file order assumed to be date order)**")
        st.table({"": ["Random split (main results)", "Time-ordered: train on first 60%, test on last 20%"],
                  "Test ROC-AUC": [f"{_t['random_split_reference']['test_roc_auc']:.3f}", f"{_t['temporal']['test_roc_auc']:.3f}"],
                  "Subscription rate in test data": [ui.pct(meta['metrics_test']['prevalence']), ui.pct(_t['prevalence']['test'])]})
        st.caption(f"Subscription rate drifts across the file: {ui.pct(_t['prevalence']['train'])} in the first 60%, "
                   f"{ui.pct(_t['prevalence']['val'])} in the next 20%, {ui.pct(_t['prevalence']['test'])} in the last 20%. "
                   "ROC-AUC does not depend on the base rate, so it is comparable across the two rows; PR-AUC is not.")

with tabs[3]:
    st.markdown("""
- **Fairness.** Age and marital status are included as inputs. Removing them lowers PR-AUC slightly (see the ablation in the report), so they carry some signal, but a customer should never be excluded from an offer solely because of a sensitive attribute. A group-level fairness audit has not been performed and is listed as future work.
- **Privacy.** The app stores nothing: uploaded files are processed in memory. Use only data you are entitled to process.
- **Transparency.** Every prediction comes with an explanation and a plain statement that it is an estimate.
- **Human oversight.** Predictions are decision support. A person should review who is contacted, and customers should be able to decline contact.
- **Marketing pressure.** Ranking customers by likelihood to accept can concentrate calls on some people. Contact frequency limits should still apply.
""")

with tabs[4]:
    st.markdown("""
**Implemented:** calibration comparison, validation-based threshold selection, bootstrap confidence intervals, lift analysis, feature ablation.

**Proposed, not implemented:** fairness audit across age and marital groups; a time-ordered evaluation if dates become available; cost-sensitive thresholds from real call costs and deposit values; a randomised pilot to measure true uplift; drift monitoring on input distributions and score calibration; robustness tests against noisy inputs.
""")

with tabs[5]:
    env = meta["environment"]
    st.markdown("Versions below were recorded when the models were trained.")
    st.table({"Package": ["Python", "scikit-learn", "imbalanced-learn", "pandas", "NumPy", "SHAP", "joblib", "Streamlit", "Plotly"],
              "Version": [env["python"], env["scikit-learn"], env["imbalanced-learn"], env["pandas"], env["numpy"],
                          env["shap"], env["joblib"], st.__version__, __import__("plotly").__version__]})
    st.caption(f"Random seed {meta['training_config']['seed']}. Trained {meta['created_utc']} (UTC). "
               f"Training time {meta['train_seconds']} s.")
