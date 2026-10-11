# BankSmart AI- Bank Term Prediction model

Predict which bank customers are likely to subscribe to a term deposit when contacted by phone, and explain why.
An end-to-end machine-learning project: data validation, leakage-aware preprocessing, model comparison, probability
calibration, threshold selection, SHAP explanations, campaign lift analysis and a 7-page Streamlit dashboard.
Runs fully locally: no paid APIs, no cloud accounts, no secrets.

> **Read this first.** All results are *offline* measurements on one historical dataset (a Portuguese bank,
> 2008-2010). They do not show that using the model would create extra subscriptions, and no revenue or ROI claim is
> made. The dataset is ordered by date but the main evaluation uses a random split, so the headline numbers are
> optimistic for forecasting a future campaign (see *Limitations*, and `reports/FINAL_REPORT.md`).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<Nischay_S>/<Bank-Term-Deposit>/blob/main/notebooks/BankSmart_AI_Colab.ipynb)


## Two scenarios

| Scenario | Inputs | Use |
|---|---|---|
| **Pre-call** | everything known before dialling (no call duration) | deciding whom to call |
| **Post-call** | adds call duration | understanding conversion only; duration is unknown before the call |

Separate models and artifacts are kept for each (`artifacts/pre_call/`, `artifacts/post_call/`).

## Dataset

Source: UCI Machine Learning Repository, *Bank Marketing* (dataset 222), CC BY 4.0.
Moro, S., Rita, P., & Cortez, P. (2014). https://doi.org/10.24432/C5K306

This project uses the 17-column `bank-full.csv` (45,211 rows). Two ways to get it:

* **UCI:** https://archive.ics.uci.edu/dataset/222/bank-marketing -> download `bank.zip` -> use `bank-full.csv`.
* **Kaggle:** "Direct Marketing Campaigns for Bank Term Deposits" -> `train.csv` is the same 45,211-row file.

Copy it to **`data/raw/bank_marketing.csv`** (comma or semicolon separated both work).

> **Do not use the Kaggle `test.csv` (4,521 rows) or UCI `bank.csv` for evaluation.** Every one of its rows also
> appears in the full file, so scoring on it would test on training data. The project makes its own split.

Check the file is right: `python -m pytest tests/test_data.py -q` (the real-data tests expect 45,211 rows,
39,922 "no" and 5,289 "yes").

## Regenerate the models and results

Only needed if you change the code or data. Run from the project root, in this order:

```powershell
python -m src.train --scenario all     # models, CV, calibration, threshold, metrics  (several minutes)
python -m src.explain --scenario all   # SHAP, permutation importance, logistic reference
python -m src.campaign                 # lift tables and gains curves
python -m src.summary                  # dataset summary, demo customers, explainer background
python -m src.ablation                 # optional: feature-ablation study (pre-call)
python -m src.temporal_check           # optional: time-ordered sensitivity check
```

`python -m src.train --scenario pre_call --fast` runs a quick smoke version. The app never retrains on launch.
Random seed (42), package versions and the full training configuration are saved in
`artifacts/<scenario>/metadata.json`.

## Tests

```powershell
python -m pytest -q
```

Tests that need the real dataset or trained artifacts are skipped (not failed) if those files are absent.
App pages are tested with Streamlit's headless test harness; file *uploads* are not testable that way, so the
upload parser is unit-tested directly.

## Project structure

```
app.py                      Streamlit entry point (navigation, sidebar scenario switch)
config.py                   paths, seed, schema, scenarios, thresholds, limits
requirements.txt
pages/                      the 7 dashboard pages
src/data_loader.py          loading, schema/target validation, stratified split, safe CSV upload parser
src/preprocessing.py        leakage-aware feature engineering + encoding pipeline
src/train.py                CV comparison, tuning, selection, calibration, threshold, artifacts
src/evaluate.py             metrics, threshold sweep, calibration, bootstrap CIs
src/predict.py              artifact loading, strict validation, single + batch prediction
src/explain.py              SHAP (+ fallbacks), permutation importance, plain-language explanations
src/campaign.py             top-k% lift analysis (offline)
src/ablation.py             feature-ablation study
src/temporal_check.py       time-ordered sensitivity check
src/summary.py              app-ready dataset summary, demo + background samples
src/ui.py                   Streamlit styling, cached loaders, chart helpers
artifacts/                  trained models, metrics, explanation and summary files
reports/                    final report, synopsis, slides outline, viva questions, source guide
notebooks/                  Google Colab notebook (self-contained)
tests/                      automated tests
data/raw/                   put bank_marketing.csv here (git-ignored)
```

## How the main safeguards work

* **No leakage:** `duration` is dropped by the pre-call pipeline itself, and a test enforces it. Preprocessing,
  scaling, encoding and SMOTE all run *inside* each cross-validation fold.
* **No peeking:** models, calibration and the decision threshold are chosen on validation data. The test split is
  scored afterwards.
* **Strict inputs:** missing columns raise a clear error (near-matches are suggested, never applied). Non-numeric or
  out-of-range values make a row invalid. Missing cells and unseen categories are scored but flagged.
* **Honest explanations:** SHAP contributions describe model behaviour, not causes.



   Check the list: `data/raw/bank_marketing.csv`, `venv/` and `__pycache__/` must **not** appear (they are ignored).
   `artifacts/`, `notebooks/`, `reports/`, `src/`, `pages/`, `tests/` should.


**Later changes:** `git add .` then `git commit -m "describe the change"` then `git push`.

**Good practice:** add a short repository description and topics (for example `machine-learning`, `streamlit`,
`explainable-ai`) via the gear icon on the repo page; add a `LICENSE` file if you want others to reuse the code
(your choice; the dataset itself is CC BY 4.0 and must be credited, which the README does); never commit passwords,
tokens or private data.

**Common Git problems:** *"fatal: not a git repository"* -> run `git init` or `cd` into the project folder.
*"remote origin already exists"* -> `git remote set-url origin <url>`. *"failed to push some refs"* -> you ticked
"Add a README" when creating the repo; run `git pull origin main --allow-unrelated-histories`, then push again.
*"Authentication failed"* -> use the browser sign-in or a personal access token (step 7). *File larger than 100 MB* -> not
applicable here (the largest file is about 1.5 MB).

## Optional: deploy on Streamlit Community Cloud (free)

Per Streamlit's current documentation, the app must be in a GitHub repository, with `requirements.txt` in the
repository root, and the app runs with the repository root as its working directory (this project already does).

1. Push the project to GitHub (above). Use a public repository, or grant Streamlit access to a private one.
2. Go to https://share.streamlit.io , sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main` and main file `app.py`.
4. Open **Advanced settings** and choose the **same Python version you used locally (3.12)**; Streamlit's docs warn
   that library versions depend on the Python version.
5. Deploy. No secrets are needed. The dataset is not in the repository, so retraining is not possible on the cloud,
   but the app does not need it. Cloud limits and defaults can change; check the current Streamlit docs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `py` not recognised | Reinstall Python and tick "Add python.exe to PATH"; reopen PowerShell. |
| Activation blocked ("running scripts is disabled") | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `pip install` fails building numpy/pandas/scikit-learn | Your Python is too old. Use 3.11 or newer (3.12 tested). |
| `ModuleNotFoundError: No module named 'src'` or `config` | Run commands from the project root (the folder containing `app.py`). |
| "Trained model files were not found" | Run `python -m src.train --scenario all`, or make sure the `artifacts/` folder was copied. |
| `FileNotFoundError: Dataset not found` | Put the CSV at `data/raw/bank_marketing.csv`. |
| Error loading `model.joblib` mentioning scikit-learn versions or pickle | Models are tied to scikit-learn 1.8.0. Install the pinned `requirements.txt` or retrain. |
| Explanations say `background_sample.csv` is missing | `python -m src.summary` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Path contains spaces and a command fails | Wrap the path in quotes, or move the project to a folder like `C:\projects\banksmart-ai`. |
| Upload rejected | Read the message: it names the exact problem (extension, size, encoding, duplicate or missing columns). Save as "CSV UTF-8". |
| Everything `pytest` skips | The dataset or artifacts are not in place; see *Dataset* and *Regenerate*. |

## Limitations (short version)

Offline evaluation only; one bank and period; **time-ordered data evaluated with a random split**; no cost or
revenue data (the recall-favouring threshold is an assumption); no fairness audit yet; SMOTE applied to one-hot
columns; top models are statistically close. Full discussion in `reports/FINAL_REPORT.md`.

## Citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5K306 (CC BY 4.0). Original study: Moro, Cortez & Rita (2014), *Decision Support
Systems* 62, 22-31.
