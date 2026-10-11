# Source File Guide

A concise explanation of every major file. Run commands from the project root.

## Top level
| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point. Sets the page config, builds sidebar navigation for the seven pages, and shows the scenario switch. The Overview page is defined here. |
| `config.py` | Single source of truth: paths (via `pathlib`), seed, split sizes, schema, category domains, scenario definitions (`LEAKAGE_FEATURES`), threshold objective (`THRESHOLD_BETA`), upload limits. `scenario_features()` returns the allowed raw columns per scenario. |
| `requirements.txt` | Pinned dependencies. |
| `README.md` | Setup, run, test, GitHub and deployment instructions, troubleshooting. |
| `.streamlit/config.toml` | Theme (light, navy/teal) and upload limit. |

## `src/` (logic, no Streamlit except `ui.py`)
| File | Purpose |
|---|---|
| `data_loader.py` | `load_raw` (separator detection, schema/type/category/target validation), `split_features_target` (y -> 1/0), `make_splits` (stratified 60/20/20), `read_uploaded_csv` (safe parser: extension, size, rows, encoding, duplicate and missing headers). Raises `DataValidationError`. |
| `preprocessing.py` | `FeatureEngineer` (selects scenario columns, raises `FeatureMismatchError` on missing ones, drops extras, builds `previously_contacted`, `negative_balance`) and `build_preprocessor` (imputation, scaling, one-hot with unseen categories ignored). |
| `train.py` | Builds candidates (SMOTE inside the pipeline), tunes forest and boosting, cross-validates on the training split, scores validation and test identically, selects by validation PR-AUC, compares calibration methods, chooses the threshold on validation, computes test metrics and bootstrap CIs, saves artifacts and metadata (seed, versions, configuration). CLI: `python -m src.train --scenario all`. |
| `evaluate.py` | Metrics (precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, Brier, log-loss, ECE), exact F-beta threshold search, threshold sweep, reliability curve, class report, bootstrap CIs. |
| `predict.py` | `load_bundle` (loads model, base model, metadata; raises `ArtifactsMissingError` with the training command), `validate_frame` (strict per-row validation), `predict_frame` (batch) and `predict_single`. Never substitutes columns or invents values. |
| `explain.py` | `ModelExplainer` (SHAP TreeExplainer, LinearExplainer, or occlusion fallback), grouped local explanations, plain-language `describe_explanation`, global SHAP, permutation importance, logistic reference, SHAP summary figure, artifact generation (`python -m src.explain`). |
| `campaign.py` | Ranking, top-k% table (selected, found, precision, recall, lift, random baseline), gains curve, `select_top_fraction`; includes the offline disclaimer. |
| `ablation.py` | Refits the selected architecture without feature groups and reports test PR-AUC change with paired bootstrap intervals. |
| `temporal_check.py` | Supplementary time-ordered split to quantify how optimistic the random split is. |
| `summary.py` | Writes `dataset_summary.json`, `background_sample.csv`, `demo_customers.csv` so the app runs without the raw dataset. |
| `ui.py` | Styling, scenario handling, cached loaders (`st.cache_resource` / `st.cache_data`), chart styling, probability gauge. |
| `build_report.py` | Fills the Markdown templates in `reports/templates/` from saved artifacts; fails on unknown tokens. |

## `pages/`
| File | Purpose |
|---|---|
| `1_Executive_Dashboard.py` | Headline test metrics with CIs, dataset facts, subscription rates, model comparison, pre/post comparison. |
| `2_Customer_Prediction.py` | Validated form, adjustable threshold, probability gauge, SHAP explanation and disclaimer. |
| `3_Batch_Prediction.py` | CSV upload, validation report, results table, summary, CSV download. |
| `4_Model_Performance.py` | Comparison tables, confusion matrix, ROC/PR curves, threshold tuning, calibration, cross-validation. |
| `5_Explainable_AI.py` | Global importance, SHAP summary, cross-checks, individual explanations. |
| `6_Campaign_Analytics.py` | Offline lift analysis and ranking of newly scored customers, clearly separated. |
| `7_About_Project.py` | Problem, method, limitations (with the temporal check), ethics, future scope, recorded versions. |

## `notebooks/`
| File | Purpose |
|---|---|
| `BankSmart_AI_Colab.ipynb` | Self-contained Google Colab version: uploads the dataset, validates it, trains and compares the eight models, calibrates, picks the threshold, evaluates on the test set with bootstrap CIs, explains with SHAP (with fallback), computes campaign lift, predicts single and batch customers, saves the model, and runs a time-ordered check. Same seed and split procedure as the project. |

## `tests/`
| File | Covers |
|---|---|
| `conftest.py` | Synthetic fixtures (for logic only) and fake artifacts. |
| `test_data.py` | Loading, schema and target validation, splits, real-data checks. |
| `test_upload.py` | Safe upload parsing. |
| `test_preprocessing.py` | Leakage exclusion, unseen categories, missing values, mismatch errors. |
| `test_training.py` | Training wiring on synthetic data, artifacts, metadata, SMOTE placement. |
| `test_prediction.py` | Artifact loading, single/batch prediction, thresholds, validation, reproduction of saved predictions. |
| `test_explain_campaign.py` | SHAP additivity, fallbacks, wording, lift arithmetic. |
| `test_app.py` | Every page in both scenarios, form submit, demo batch, campaign ranking, missing artifacts. |

## `artifacts/`
Per scenario (`pre_call/`, `post_call/`): `model.joblib` (calibrated pipeline), `base_model.joblib` (uncalibrated, for explanations), `metadata.json`, comparison / CV / calibration / threshold CSVs,
predictions, `explain/` files, campaign and ablation tables. Shared: `dataset_summary.json`, `background_sample.csv`, `demo_customers.csv`.
