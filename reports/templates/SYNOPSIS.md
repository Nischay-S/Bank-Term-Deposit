# Project Synopsis

**Title:** BankSmart AI: Predicting Term-Deposit Subscription in Bank Telemarketing with Explainable Machine Learning
**Submitted by:** *[Name(s), roll number(s)]*  **Guide:** *[Name, designation]*  **Department:** *[Department, college]*  **Year:** *[YYYY-YY]*

## 1. Introduction
Banks phone customers to offer term deposits, but most calls do not lead to a subscription. A model that estimates each customer's
probability of subscribing can help prioritise calls, provided it is evaluated honestly and uses only information available before the call.

## 2. Problem statement
Estimate the probability that a contacted customer subscribes to a term deposit, using only legitimately available information, evaluate it with
metrics suited to a rare positive class (about @@PREV@@ of contacts), explain each prediction, and support campaign prioritisation.

## 3. Objectives
1. Validate the dataset and detect problems in the supplied train/test files.
2. Build separate **pre-call** and **post-call** models, with call duration excluded from the pre-call model.
3. Compare a baseline with imbalance-aware approaches and several model families using cross-validation.
4. Select the model, calibration and threshold on validation data; evaluate once on a held-out test set with confidence intervals.
5. Explain predictions (SHAP) without implying causation.
6. Analyse campaign prioritisation (lift) as offline evidence only.
7. Deliver a tested, reproducible Streamlit application that runs locally with no paid services.

## 4. Scope
In scope: the public UCI Bank Marketing data (@@N_ROWS@@ contacts, 16 input features), offline modelling and evaluation, a seven-page web application.
Out of scope: live deployment in a bank, measurement of real revenue or uplift, personal data processing.

## 5. Methodology
Stratified 60/20/20 split; leakage-aware preprocessing inside the model pipeline; @@CV_FOLDS@@-fold cross-validation of eight candidate models (logistic regression
plain / class-weighted / SMOTE, decision tree, random forest, gradient boosting, tuned forest and boosting); selection by validation PR-AUC; calibration (none, sigmoid,
isotonic) chosen by validation Brier score; threshold chosen on validation data by maximising F2; one test evaluation with 1,000-resample bootstrap intervals; SHAP,
permutation importance, lift analysis, feature ablation and a time-ordered sensitivity check.

## 6. Tools and technologies
Python @@ENV_PY@@, scikit-learn @@ENV_SK@@, imbalanced-learn @@ENV_IMB@@, pandas, NumPy, SHAP @@ENV_SHAP@@, Matplotlib, Plotly, Streamlit, joblib, pytest, Git/GitHub.

## 7. Results obtained (test set, from saved artifacts)
| | Pre-call | Post-call |
|---|---|---|
| Selected model | @@PRE_MODEL@@ | @@POST_MODEL@@ |
| PR-AUC (95% CI) | @@PRE_PRAUC@@ (@@PRE_PRAUC_CI@@) | @@POST_PRAUC@@ (@@POST_PRAUC_CI@@) |
| ROC-AUC | @@PRE_ROC@@ | @@POST_ROC@@ |
| Recall / precision at chosen threshold | @@PRE_REC@@ / @@PRE_PREC@@ | @@POST_REC@@ / @@POST_PREC@@ |

A supplementary time-ordered evaluation gives a pre-call test ROC-AUC of @@TEMP_TEST_ROC@@ (versus @@TEMP_RAND_ROC@@ under the random split), so the headline figures are optimistic for forecasting a later period.

## 8. Genuine contributions beyond a basic notebook
Leakage control enforced by the pipeline and tests; detection that the supplied `test.csv` overlaps the training file; validation-only selection with bootstrap CIs; a time-ordered
sensitivity analysis; feature ablation with paired intervals; verified SHAP explanations with fallbacks; strict batch-input handling; offline campaign analytics with no unsupported ROI claims;
recorded seeds and versions; a template-driven report with no hand-typed numbers; an automated test suite.

## 9. Limitations
Offline only; one bank and period; time-ordered data evaluated with a random split; no cost or revenue data; no fairness audit; light hyper-parameter tuning.

## 10. Work plan *(adjust to your calendar)*
| Phase | Activity |
|---|---|
| 1 | Literature survey, dataset study, problem definition |
| 2 | Data validation, preprocessing, leakage controls |
| 3 | Model training, cross-validation, tuning, evaluation |
| 4 | Calibration, thresholding, explainability, campaign analysis |
| 5 | Application development and testing |
| 6 | Documentation, report and presentation |

## 11. References
1. Moro, Cortez, Rita (2014), *Decision Support Systems* 62:22-31.
2. Moro, Rita, Cortez, *Bank Marketing* [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5K306.
3. Saito & Rehmsmeier (2015), *PLoS ONE* 10(3):e0118432.
4. Lundberg & Lee (2017), NeurIPS 30.
5. Pedregosa et al. (2011), *JMLR* 12:2825-2830.
*(Full list in the final report.)*
