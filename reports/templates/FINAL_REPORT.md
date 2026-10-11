# BankSmart AI: Predicting Term-Deposit Subscription in Bank Telemarketing with Explainable Machine Learning

*Final-year project report (Markdown source; convert to Word/PDF with your college template).*
*All numerical results in this document are generated from the saved training artifacts by `python -m src.build_report`. Nothing is typed by hand.*

---

## 1. Title page (template)

| | |
|---|---|
| **Project title** | BankSmart AI: Predicting Term-Deposit Subscription in Bank Telemarketing with Explainable Machine Learning |
| **Submitted by** | *[Student name(s), roll number(s)]* |
| **Programme / branch** | *[B.Tech / B.E. in ..., Artificial Intelligence and Machine Learning]* |
| **Guide** | *[Name, designation]* |
| **Department / institution** | *[Department, college, university]* |
| **Academic year** | *[YYYY-YY]* |
| **Date of submission** | *[Date]* |

---

## 2. Abstract

Banks use phone campaigns to sell term deposits, but most calls end without a subscription. This project builds
BankSmart AI, a reproducible machine-learning system that estimates the probability that a contacted customer
subscribes, explains each estimate, and supports campaign prioritisation. It uses the public UCI Bank Marketing
data (@@N_ROWS@@ contacts, @@PREV@@ subscribed). Two scenarios are modelled separately: a **pre-call** scenario that uses
only information available before dialling, and a **post-call** scenario that also uses call duration, which is
unknown until the call ends. Eight candidate models, including imbalance-aware variants, were compared by
@@CV_FOLDS@@-fold cross-validation; the model, probability calibration and decision threshold were chosen on a separate
validation set, and a held-out test set was scored once afterwards. For the pre-call scenario the selected
@@PRE_MODEL@@ model reached a test PR-AUC of @@PRE_PRAUC@@ (95% bootstrap CI @@PRE_PRAUC_CI@@) and ROC-AUC of @@PRE_ROC@@; at its
validation-chosen threshold it found @@PRE_REC@@ of subscribers with a precision of @@PRE_PREC@@. Adding call duration raises test
PR-AUC by @@DUR_GAIN_PR@@ and ROC-AUC by @@DUR_GAIN_ROC@@, but that information is not available when deciding whom to call. SHAP
explanations, a feature-ablation study, a lift analysis and a seven-page Streamlit application are provided. A
supplementary time-ordered evaluation shows that the headline results are optimistic for forecasting a later period
(test ROC-AUC @@TEMP_TEST_ROC@@ versus @@TEMP_RAND_ROC@@ under the random split), so the results are reported as offline evidence only; no
revenue, ROI or causal uplift claim is made.

**Keywords:** class imbalance, probability calibration, SHAP, precision-recall, telemarketing, term deposit, Streamlit.

---

## 3. Introduction

Direct marketing by telephone remains common in retail banking. Each call costs agent time and can irritate the
customer, while only a small share of calls succeed. If a bank can rank customers by how likely they are to say
yes, it can concentrate effort where it is most useful and avoid contacting people who are very unlikely to accept.
This is a binary classification problem with a rare positive class, so the choice of evaluation metrics, the handling
of class imbalance, the calibration of probabilities and the treatment of information that is only known after the
outcome (leakage) matter as much as the choice of algorithm.

## 4. Problem statement

Given the attributes of a bank customer and the history of contact with them, estimate the probability that the
customer will subscribe to a term deposit when contacted by phone, in a form that (a) uses only information
legitimately available at the time of the decision, (b) is evaluated honestly with metrics suited to a rare positive
class, (c) can be explained to a human reviewer, and (d) can be used to rank customers for outreach.

## 5. Motivation and significance

- **Cost asymmetry.** Wasted calls and missed subscribers have different costs; a single accuracy number hides this.
  A model that predicts "no" for everyone is @@ALWAYS_NO_ACC@@ accurate on this data and useless.
- **Leakage is easy to miss.** Call duration is strongly related to the outcome but is only known after the call. A
  model that uses it looks excellent and cannot be deployed for choosing whom to call.
- **Trust.** Staff and customers are more likely to accept decision support that can say why a customer was ranked
  highly, and that states its limits.
- **Educational value.** The project exercises the full workflow: data validation, leakage control, model selection,
  calibration, thresholding, explainability, testing and deployment.

## 6. Objectives

1. Validate the dataset and target, and detect problems with the supplied train/test files.
2. Build leakage-aware pipelines for a pre-call and a post-call scenario.
3. Compare a baseline with imbalance-aware approaches and several model families using cross-validation.
4. Select the model, calibration and threshold on validation data; report a single final test evaluation with confidence intervals.
5. Explain predictions globally and per customer, distinguishing association from causation.
6. Analyse campaign prioritisation (precision, recall and lift in the top 10/20/30%) as offline evidence.
7. Deliver a tested, reproducible, documented Streamlit application that runs without paid services.
8. State limitations honestly, including time ordering of the data.

---

## 7. Literature review

*Only sources that exist are cited. Items [1] and [2] were checked against the UCI repository page; please confirm the
volume and page details of every entry in Google Scholar or the publisher site before final submission.*

**The data and its original study.** The data come from direct phone campaigns of a Portuguese bank and were
analysed by Moro, Cortez and Rita [1], who proposed a data-driven approach to predicting telemarketing success.
An earlier CRISP-DM-based study of the same domain is given in [2]. The dataset documentation itself notes that call
duration is not known before a call, so it should be excluded in any realistic predictive model; this project builds
that rule into its pipeline.

**Imbalanced classification and evaluation.** When positives are rare, accuracy is misleading. Saito and Rehmsmeier
[3] argue that the precision-recall plot is more informative than the ROC plot on imbalanced data, and Davis and
Goadrich [4] relate the two curves. Fawcett [5] gives a standard introduction to ROC analysis. Chawla et al. [6]
introduced SMOTE, which creates synthetic minority examples by interpolation; it must be applied only to training
data, which is why it sits inside the cross-validated pipeline here.

**Models.** Logistic regression and decision trees are described in [7]. Random forests [8] average many decorrelated
trees, and gradient boosting [9] builds an additive model of trees by fitting residual gradients. Scikit-learn [10]
supplies the implementations used here.

**Probability calibration.** Sigmoid (Platt) scaling [11] and isotonic regression [12] map classifier scores to
probabilities; Niculescu-Mizil and Caruana [13] compare them empirically. The Brier score [14] and expected
calibration error [15] measure calibration quality.

**Explainability.** Shapley values [16] from cooperative game theory underlie SHAP [17]; TreeSHAP computes them
exactly and efficiently for tree ensembles [18].

**Uncertainty, fairness and drift.** The bootstrap [19] provides confidence intervals without distributional
assumptions. Equality-of-opportunity criteria [20] frame group fairness, and Gama et al. [21] survey concept drift;
both are discussed as future work here.

---

## 8. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, Bank Marketing (dataset 222); CC BY 4.0; DOI 10.24432/C5K306 |
| File used | `bank-full.csv` (17 columns), obtained via a Kaggle copy named *Direct Marketing Campaigns for Bank Term Deposits* (`train.csv`) |
| Rows | @@N_ROWS@@ |
| Subscribed (`y = yes`) | @@N_POS@@ (@@PREV@@) |
| Not subscribed | @@N_NEG@@ |
| Missing values | none; "unknown" is a category in several columns |
| Duplicate rows | none |

**Inputs (16):** age, job, marital, education, default (credit in default), balance, housing (housing loan), loan
(personal loan), contact (channel), day, month, duration (seconds, last call), campaign (contacts in this campaign),
pdays (days since previous campaign contact; -1 = never), previous (contacts before this campaign), poutcome
(outcome of previous campaign). **Target:** `y`.

**Data notes.** @@PDAYS_NEVER@@ of customers were never contacted before (`pdays = -1`), so `poutcome` is "unknown" for
@@UNK_POUTCOME@@; the contact channel is "unknown" for @@UNK_CONTACT@@ and education for @@UNK_EDU@@.

Subscription rate by month (the data span May 2008 to November 2010 according to the UCI documentation):

@@TABLE_MONTH_RATES@@

Subscription rate by outcome of the previous campaign:

@@TABLE_POUTCOME_RATES@@

**The supplied test file.** The Kaggle copy also contains `test.csv` (4,521 rows). Every one of its rows also
appears exactly in `train.csv`, so evaluating on it would test on training data. It was not used. A fresh stratified
split was made instead.

**Date ordering.** The UCI documentation states the file is ordered by date. The month sequence in the file agrees: it forms
@@TEMP_MONTH_RUNS@@ consecutive blocks that progress through the calendar (first blocks: @@TEMP_RUNS_HEAD@@). The project brief instructed treating the rows as
randomly ordered, so the main evaluation uses a random split; the consequences are tested in section 17.4.

---

## 9. System requirements

**Software** (versions recorded at training time): Python @@ENV_PY@@, scikit-learn @@ENV_SK@@, pandas @@ENV_PD@@, NumPy @@ENV_NP@@,
imbalanced-learn @@ENV_IMB@@, SHAP @@ENV_SHAP@@, plus Streamlit, Plotly, Matplotlib, joblib and pytest (pinned in `requirements.txt`).
**Hardware:** any modern laptop; the full training run took about @@PRE_TRAIN_SECONDS@@ s (pre-call) and @@POST_TRAIN_SECONDS@@ s (post-call)
on a single CPU core with 3 GB RAM. **Network:** only needed to install packages and download the dataset.
**Accounts / cost:** none; no paid API, external LLM or cloud account.

## 10. System architecture

```
 raw CSV --> data_loader (schema + target validation, stratified split)
                |
                v
        preprocessing (FeatureEngineer -> impute/scale/one-hot)  <-- scenario decides allowed columns
                |
                v
   train (CV comparison, tuning, selection on validation, calibration, threshold)
                |
                v                                   artifacts/<scenario>/  (model, metadata, tables)
   evaluate  <---------------------------------------------------------------+
                |                                                             |
   predict (strict validation, single + batch)    explain (SHAP, fallbacks)   campaign (lift)
                \                |                         /
                 `---------  Streamlit app (7 pages)  ----'
```

Design rules: training is separate from app start-up; the app only loads saved artifacts; paths are built with
`pathlib` relative to the project root; prediction and explanation use the same saved pipeline; pre-call and post-call
artifacts live in separate folders and are checked against each other when loaded.

---

## 11. Methodology

1. **Validate** the file (columns, types, category values, target values and presence of both classes).
2. **Split** the rows with stratification into training (60%, @@N_TRAIN@@), validation (20%, @@N_VAL@@) and test (20%, @@N_TEST@@), seed @@SEED@@.
3. **Preprocess inside the model pipeline** so that statistics (medians, scaling, category levels) are learned from
   training folds only.
4. **Compare** candidate models by @@CV_FOLDS@@-fold stratified cross-validation on the training split.
5. **Fit** each candidate on the training split and score validation and test splits identically.
6. **Select** the model by highest validation PR-AUC, then **calibrate** (none, sigmoid, isotonic) and pick by validation Brier score.
7. **Choose the threshold** on validation probabilities by maximising F2 (assumption: a missed subscriber costs more than an extra call).
8. **Evaluate once** on the test split; compute bootstrap confidence intervals.
9. **Explain** the selected model; **analyse** campaign lift; run ablation and temporal checks.

## 12. Mathematical foundations

### 12.1 Logistic regression and the sigmoid

For inputs $x\in\mathbb{R}^p$ with weights $w$ and bias $b$, the model estimates

$$P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}},\qquad z=w^\top x+b .$$

The sigmoid maps any real score to $(0,1)$. Weights minimise the (L2-regularised) negative log-likelihood

$$\mathcal{L}(w,b)=-\sum_{i}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]+\tfrac{1}{2C}\lVert w\rVert^2 .$$

$e^{w_j}$ is the multiplicative change in the odds of $y=1$ per unit increase of feature $j$, other features fixed.

### 12.2 Decision trees and random forests

A tree splits the data to reduce impurity. For a node with positive fraction $p$ the Gini impurity is
$G=2p(1-p)$ and the entropy is $H=-p\log_2 p-(1-p)\log_2(1-p)$; the split chosen maximises the impurity decrease
$\Delta=G_{\text{parent}}-\sum_{c}\tfrac{n_c}{n}G_c$. A random forest [8] averages $B$ trees, each trained on a bootstrap sample
and a random subset of features per split: $\hat p(x)=\tfrac1B\sum_{b=1}^{B}\hat p_b(x)$. Averaging reduces variance.

### 12.3 Gradient boosting

Gradient boosting [9] builds $F_m(x)=F_{m-1}(x)+\nu\,h_m(x)$, where the small tree $h_m$ is fitted to the negative
gradient of the loss at $F_{m-1}$ and $\nu$ is the learning rate. For log-loss the gradient is $y-\sigma(F_{m-1}(x))$.
The model used here (scikit-learn's histogram-based implementation) bins features into at most 255 bins to speed up split finding.

### 12.4 The confusion matrix and derived metrics

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actual positive** | TP | FN |
| **Actual negative** | FP | TN |

$$\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2 P+R},$$
$$\text{Balanced accuracy}=\tfrac12\Big(\tfrac{TP}{TP+FN}+\tfrac{TN}{TN+FP}\Big),\qquad \text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.$$

$F_1$ is the harmonic mean of precision and recall. $F_2$ ($\beta=2$) weights recall more than precision.

### 12.5 ROC-AUC and PR-AUC

The ROC curve plots recall (true-positive rate) against the false-positive rate $FP/(FP+TN)$ over all thresholds;
ROC-AUC equals the probability that a random positive is scored above a random negative. The PR curve plots
precision against recall; PR-AUC is summarised here by average precision, $\text{AP}=\sum_k (R_k-R_{k-1})P_k$.
A random classifier has PR-AUC equal to the prevalence (@@PREV@@), whereas its ROC-AUC is 0.5 regardless of prevalence,
which is why PR-AUC is more sensitive to performance on a rare class [3].

### 12.6 Class imbalance

With positive share $\pi$ (about @@PREV@@ here), a classifier can reach accuracy $1-\pi$ by always predicting "no". Options used here:
**class weights** $w_c=\tfrac{n}{2\,n_c}$ increase the loss for the minority class; **SMOTE** [6] creates synthetic points
$x_{\text{new}}=x_i+\lambda(x_{nn}-x_i)$, $\lambda\sim U(0,1)$, between a minority point $x_i$ and one of its nearest minority neighbours
$x_{nn}$, applied only to training folds. Because the features here include one-hot columns, interpolation yields fractional category
values, a known limitation.

### 12.7 Probability calibration

A model is calibrated if among customers given probability $p$, about a fraction $p$ subscribe. **Platt (sigmoid) scaling** [11] fits
$P(y=1\mid s)=1/(1+e^{As+B})$ to scores $s$; **isotonic regression** [12] fits a monotone step function. The **Brier score**
$\tfrac1n\sum(p_i-y_i)^2$ [14] and **expected calibration error** $\text{ECE}=\sum_b\tfrac{n_b}{n}\,\lvert \bar y_b-\bar p_b\rvert$ over probability bins [15] quantify calibration.
Calibrators are fitted with internal cross-validation on the training split only.

### 12.8 Decision thresholds

The classifier outputs a probability $\hat p$; the decision rule is $\hat y=\mathbb{1}[\hat p\ge t]$. Lowering $t$ raises recall and lowers
precision. With unit costs $c_{FP}$ and $c_{FN}$, the expected-cost-optimal threshold for a calibrated model is
$t^*=c_{FP}/(c_{FP}+c_{FN})$. No costs are available here, so $t$ is chosen by maximising $F_2$ on validation data, and the application
lets the user move it.

### 12.9 SHAP values

For a model $f$ and a customer $x$, the Shapley value of feature $j$ is
$$\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[f_{S\cup\{j\}}(x)-f_S(x)\Big],$$
the average marginal contribution of $j$ over all orderings of features. SHAP values satisfy additivity,
$f(x)=\phi_0+\sum_j\phi_j$, where $\phi_0$ is the expected model output. Intuitively, each feature gets credit for how much it moves
the prediction away from the average prediction. TreeSHAP [18] computes them exactly for tree ensembles. Here the explained output is the
log-odds, and additivity was verified numerically (maximum error in predicted probability: pre-call @@PRE_ADD_ERR@@, post-call @@POST_ADD_ERR@@).

### 12.10 Lift and campaign prioritisation

Rank customers by $\hat p$ and select the top fraction $q$ ($k=\lceil qn\rceil$ customers). Precision@$q$ is the share of selected customers who are
subscribers; recall@$q$ is the share of all subscribers found; **lift** is $\text{Precision@}q/\pi$. Random selection has lift 1 and finds on average
$k\pi$ subscribers. Lift is an offline property of a ranking; it does not measure the causal effect of contacting someone.

---

## 13. Algorithms and pseudocode

```
ALGORITHM 1: Training pipeline (per scenario)
input: validated data D, scenario s, seed
 1  features <- allowed_columns(s)                      # pre-call excludes duration
 2  (train, val, test) <- stratified_split(D, 60/20/20, seed)
 3  candidates <- baseline + imbalance-aware + tree models; tune forest and boosting on train only
 4  for each candidate c:
 5      cv_scores[c] <- stratified_k_fold(c, train)    # preprocessing/SMOTE fitted inside each fold
 6      fit c on train; score val and test with the same code
 7  best <- argmax over c of PR-AUC(val)
 8  for method in {none, sigmoid, isotonic}: fit calibrator on train; compute Brier(val)
 9  final <- calibrated model with lowest validation Brier
10  t <- argmax_t F2(val, final, t)
11  report metrics, confusion matrix, bootstrap CIs on test; save model + metadata
```

```
ALGORITHM 2: Batch prediction with strict validation
input: table T, model bundle
 1  if any required column missing: raise error naming it (suggest near-matches, never apply)
 2  for each row r:
 3      if a numeric value is non-numeric or out of range: mark r invalid, no prediction
 4      note missing cells and unseen categories on r (still scored)
 5  p <- model.predict_proba(valid rows); class <- (p >= t)
 6  return rows with probability, class, status, data-quality note
```

```
ALGORITHM 3: Campaign ranking
 1  sort customers by probability, descending
 2  for q in {10%, 20%, 30%}: k <- ceil(q*n); found <- positives among top k
 3  precision <- found/k; recall <- found/total_positives; lift <- precision/prevalence
```

---

## 14. Data preprocessing

- **Schema and category checks** at load; **target** must be exactly `yes`/`no` with both present.
- **Scenario columns.** Pre-call excludes `duration`; extra columns in uploaded data are ignored and reported.
- **Derived features:** `previously_contacted` (from `pdays != -1`), `pdays` set to 0 when never contacted, `negative_balance` flag.
- **Imputation:** numeric median, categorical constant "missing" (training data has no missing cells; this protects at prediction time).
- **Scaling** (linear models only) and **one-hot encoding** with unseen categories mapped to all zeros.
- **Leakage rule:** every learned statistic is fitted inside the training fold. `campaign` (contacts including the current one) is borderline
  pre-call information; an ablation (section 17.5) measures how much it matters.

## 15. Experimental setup

| Setting | Value |
|---|---|
| Split | stratified random 60/20/20, seed @@SEED@@ |
| Cross-validation | @@CV_FOLDS@@-fold stratified, on training split only |
| Candidates | logistic regression (plain, class-weighted, SMOTE), decision tree (depth 6, balanced), random forest (balanced), histogram gradient boosting, tuned forest, tuned boosting |
| Tuning | randomised search, 5 candidates, 3-fold inner CV, scoring average precision (deliberately light) |
| Selection | highest validation PR-AUC |
| Calibration | none / sigmoid / isotonic, internal 5-fold CV on training split; lowest validation Brier |
| Threshold | maximise F2 on validation probabilities (own threshold per model for the comparison tables) |
| Confidence intervals | 1,000 bootstrap resamples of the test set |
| Tuning result (pre-call) | @@PRE_TUNE@@ |

---

## 16. Model evaluation

### 16.1 Cross-validation (pre-call, training split)

@@TABLE_CV_PRE@@

Recall and precision here use the default 0.5 cut-off, so class-weighted models look very different from unweighted ones; PR-AUC and
ROC-AUC do not depend on a cut-off.

### 16.2 Validation comparison (pre-call; each model at its own validation-chosen threshold)

@@TABLE_COMP_VAL_PRE@@

### 16.3 Test comparison (pre-call; reported for transparency, not used for selection)

@@TABLE_COMP_TEST_PRE@@

### 16.4 Calibration (pre-call, validation)

@@TABLE_CAL_PRE@@

Chosen calibration: **@@PRE_CAL@@** (lowest validation Brier score).

---

## 17. Results and discussion

### 17.1 Selected models

| | Pre-call | Post-call |
|---|---|---|
| Selected model | @@PRE_MODEL@@ | @@POST_MODEL@@ |
| Calibration | @@PRE_CAL@@ | @@POST_CAL@@ |
| Threshold | @@PRE_THR@@ | @@POST_THR@@ |
| Test PR-AUC (95% CI) | @@PRE_PRAUC@@ (@@PRE_PRAUC_CI@@) | @@POST_PRAUC@@ (@@POST_PRAUC_CI@@) |
| Test ROC-AUC (95% CI) | @@PRE_ROC@@ (@@PRE_ROC_CI@@) | @@POST_ROC@@ (@@POST_ROC_CI@@) |
| Recall (95% CI) | @@PRE_REC@@ (@@PRE_REC_CI@@) | @@POST_REC@@ (@@POST_REC_CI@@) |
| Precision (95% CI) | @@PRE_PREC@@ (@@PRE_PREC_CI@@) | @@POST_PREC@@ (@@POST_PREC_CI@@) |
| F1 / balanced accuracy | @@PRE_F1@@ / @@PRE_BAL@@ | @@POST_F1@@ / @@POST_BAL@@ |
| Accuracy | @@PRE_ACC@@ | @@POST_ACC@@ |
| Brier / ECE | @@PRE_BRIER@@ / @@PRE_ECE@@ | @@POST_BRIER@@ / @@POST_ECE@@ |
| TP / FP / FN / TN | @@PRE_TP@@ / @@PRE_FP@@ / @@PRE_FN@@ / @@PRE_TN@@ | @@POST_TP@@ / @@POST_FP@@ / @@POST_FN@@ / @@POST_TN@@ |

Pre-call class-wise report (test):

@@TABLE_CLASSREP_PRE@@

**Why not accuracy?** Predicting "no" for everyone scores @@ALWAYS_NO_ACC@@ accuracy and finds no subscribers. The selected pre-call model's accuracy
(@@PRE_ACC@@) is @@ACC_REL@@ that always-no baseline of @@ALWAYS_NO_ACC@@, because the threshold deliberately flags many customers. Accuracy alone would therefore misjudge it, which is why recall, precision, PR-AUC and balanced accuracy are reported instead.

**False positives versus false negatives.** At the chosen pre-call threshold the model makes @@PRE_FP@@ false positives (wasted calls) and @@PRE_FN@@ false
negatives (missed subscribers) on the test set. The F2 objective deliberately accepts more of the first to reduce the second. That is an
assumption about relative costs; with real call costs and deposit values the threshold should be recomputed.

### 17.2 Imbalance handling

@@IMBALANCE_FINDING@@

### 17.3 Model ranking

The three best pre-call models on validation PR-AUC were @@TOP_MODELS@@. The gap between first and third is @@TOP_GAP@@, compared with a
cross-validation standard deviation of about @@CV_STD_SEL@@ for the selected model, so the ordering among the top models should be treated as tentative.

### 17.4 Pre-call versus post-call, and the time-ordered check

@@TABLE_SCENARIOS@@

The post-call model is much stronger because call duration encodes how the conversation went. It cannot be used to decide whom to call.

**Time-ordered sensitivity check.** Training on the first 60% of rows in file order and testing on the last 20% (assuming file order is time order):

@@TABLE_TEMPORAL@@

The subscription rate drifts across the file (@@TEMP_PREV_TRAIN@@ in the first 60%, @@TEMP_PREV_VAL@@ in the next 20%, @@TEMP_PREV_TEST@@ in the last 20%), the first block contains only the months
@@TEMP_TRAIN_MONTHS@@, and validation ROC-AUC was @@TEMP_VAL_ROC@@. ROC-AUC does not depend on prevalence, so the drop from @@TEMP_RAND_ROC@@ to @@TEMP_TEST_ROC@@ is a fair comparison; PR-AUC is not
comparable across the two rows because the base rates differ. This check is supplementary and uses default boosting without calibration, but it
shows that **a random split overstates how well the model would forecast a later campaign**. The main results should be read with that in mind.

### 17.5 Feature ablation (pre-call)

Selected architecture refitted with feature groups removed; the last column is a paired bootstrap interval for the change in test PR-AUC.

@@TABLE_ABLATION@@

Removing `campaign` changed test PR-AUC by @@ABL_CAMPAIGN@@, @@ABL_CAMPAIGN_READ@@. Removing `day` and
`month` changed it by @@ABL_DAYMONTH@@, which @@ABL_DM_READ@@; given the date ordering and drift above, any contact-timing signal may reflect campaign waves rather than
customer traits. Removing the sensitive attributes `age` and `marital` changed it by @@ABL_SENS@@.

### 17.6 Campaign prioritisation (offline, pre-call, test set)

@@TABLE_CAMPAIGN_PRE@@

Lift is precision among selected customers divided by the @@PREV@@ base rate. **These are historical ranking results only.** They do not show that contacting these
customers would cause extra subscriptions, and no revenue, cost or ROI figure follows from them. Measuring real uplift would need a randomised pilot.

Post-call (for comparison only; not usable for choosing whom to call):

@@TABLE_CAMPAIGN_POST@@

---

## 18. Explainable AI

**Method.** SHAP (@@PRE_SHAP_METHOD@@ explainer) on the uncalibrated base model with its exact preprocessing; one-hot columns are added back together so each value belongs to one original feature.
If SHAP were unavailable, the application falls back to permutation importance and occlusion (a tested fallback). Calibration is a monotone rescaling of the same score, so contribution ranking is unchanged.

Pre-call global importance (mean |SHAP|, 2,000 validation customers):

@@TABLE_SHAP_PRE@@

Permutation importance (drop in validation PR-AUC when a feature is shuffled):

@@TABLE_PERM_PRE@@

The two methods share @@PRE_TOPK_OVERLAP@@ of their top five features (SHAP top five: @@PRE_TOP_SHAP@@; permutation top five: @@PRE_TOP_PERM@@); differences in order are expected when features are correlated.

Logistic-regression reference (an interpretable model, not the production model; numeric inputs standardised):

Largest positive coefficients:

@@TABLE_LR_POS_PRE@@

Largest negative coefficients:

@@TABLE_LR_NEG_PRE@@

**Association, model behaviour and causation.** A SHAP value says how the *model* used a feature for a prediction; it reflects associations in historical data.
It does not say that changing the feature would change the customer's behaviour. The application therefore words explanations as "raised the model's estimate", never as causes.

**Individual explanations.** For each prediction the application shows the strongest positive and negative contributions and a plain-language summary.

---

## 19. Application screenshots (to be captured after you run the app)

No screenshots are included, because none could be captured in the development environment. Capture these from your own run (`streamlit run app.py`) and insert them:

| # | Page | What to show |
|---|---|---|
| 1 | Overview | landing page with the scenario switch |
| 2 | Executive Dashboard | metric cards and class distribution |
| 3 | Executive Dashboard | model comparison chart |
| 4 | Customer Prediction | completed form, gauge and explanation |
| 5 | Batch Prediction | demo file results and summary |
| 6 | Batch Prediction | an error message for a file with a missing column |
| 7 | Model Performance | confusion matrix and classification report |
| 8 | Model Performance | ROC and PR curves |
| 9 | Model Performance | threshold tuning chart |
| 10 | Model Performance | calibration plot |
| 11 | Explainable AI | SHAP summary plot |
| 12 | Explainable AI | individual explanation |
| 13 | Campaign Analytics | gains and lift charts with the offline notice |
| 14 | About the Project | limitations tab |

---

## 20. Limitations

1. **Offline only.** Results measure ranking on historical data; they do not establish causal uplift, revenue or ROI.
2. **Time ordering.** The data are date-ordered but the main evaluation uses a random split; the time-ordered check suggests the headline numbers are optimistic for a future period, and the subscription rate drifts strongly over the file.
3. **One bank, one period** (2008 to 2010), with a campaign-wave structure that the model partly learns through month and day.
4. **No economic indicators** (this is the 17-column version) and **no cost or revenue data**; the recall-favouring threshold is an assumption.
5. **Data quality:** `poutcome` is mostly unknown; the contact channel is unknown for many customers.
6. **Tentative model ranking** among the top models; tuning was deliberately light (5 random-search candidates).
7. **SMOTE on one-hot columns** produces fractional category values.
8. **Bootstrap intervals** reflect test-set sampling only, not variation due to model fitting or time.
9. **No fairness audit** was performed; only a sensitive-attribute ablation.
10. **Explanations** describe model behaviour, not causes.
11. **Visual layout of the application was not reviewed in a browser** during development; functional behaviour is covered by automated tests.

## 21. Ethical considerations

- **Fairness.** Age and marital status are inputs. A customer should never be excluded from an offer solely because of a sensitive attribute; a group-level audit (for example equality of opportunity [20]) is needed before real use.
- **Privacy.** The application keeps uploaded data in memory only. Real deployments need consent, data minimisation and compliance with data-protection law.
- **Transparency.** Each prediction is presented as a probabilistic estimate with an explanation.
- **Human oversight.** Predictions support decisions; a person should review who is contacted, and customers must be able to decline contact.
- **Marketing pressure.** Ranking by acceptance likelihood can concentrate calls on some people; contact-frequency limits should still apply.

## 22. Future enhancements

*Proposed, not implemented:* a fairness audit across age and marital groups; time-aware validation and drift monitoring [21]; cost-sensitive thresholds from real costs; a randomised pilot to measure uplift; robustness tests under noisy inputs; additional features such as economic indicators (available in the 20-input version of the data); and per-segment calibration.
*Implemented:* calibration comparison, validation-based threshold selection, bootstrap CIs, lift analysis, feature ablation, time-ordered sensitivity check.

## 23. Conclusion

Using only information available before a call, a calibrated @@PRE_MODEL@@ model ranks customers well above chance (test ROC-AUC @@PRE_ROC@@, PR-AUC @@PRE_PRAUC@@ against a base rate of @@PREV@@), and its explanations are consistent with the model's
numerical behaviour. The post-call model looks much better but relies on information unavailable at decision time. The evaluation also revealed three data issues that
would have inflated results if ignored: leakage through call duration, an overlapping supplied test file, and time ordering that a random split hides. The project's value lies as much in those controls
and in the honest reporting of limits as in the model itself.

### Genuine contributions beyond a basic classification notebook

1. Two separately trained and separately stored scenarios with leakage enforced by the pipeline and by tests (including a mutation check that the tests fail if `duration` leaks).
2. Detection that the supplied `test.csv` is entirely contained in `train.csv`, and a clean split instead.
3. Selection, calibration and threshold choice on validation data only, with a single final test evaluation and bootstrap confidence intervals.
4. A time-ordered sensitivity analysis that quantifies how optimistic a random split is on date-ordered data.
5. Feature ablation with paired bootstrap intervals, including the borderline `campaign` feature and sensitive attributes.
6. SHAP explanations verified for additivity, with documented fallbacks and plain-language, non-causal wording.
7. Strict, user-safe input handling: no silent column substitution, no invented values, per-row status and notes.
8. Offline campaign analytics kept visibly separate from future predictions, with no unsupported ROI claims.
9. Reproducibility: seed, package versions and configuration saved with every model; a template-driven report so that no number is typed by hand.
10. An automated test suite covering data, preprocessing, training, prediction, explanation, campaign logic and the application pages.

---

## 24. References

1. S. Moro, P. Cortez and P. Rita, "A data-driven approach to predict the success of bank telemarketing," *Decision Support Systems*, vol. 62, pp. 22-31, 2014.
2. S. Moro, R. Laureano and P. Cortez, "Using data mining for bank direct marketing: an application of the CRISP-DM methodology," in *Proc. European Simulation and Modelling Conference (ESM'2011)*, 2011.
3. T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, e0118432, 2015.
4. J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd ICML*, 2006, pp. 233-240.
5. T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861-874, 2006.
6. N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321-357, 2002.
7. T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.
8. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.
9. J. H. Friedman, "Greedy function approximation: a gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189-1232, 2001.
10. F. Pedregosa et al., "Scikit-learn: machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
11. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61-74.
12. B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD*, 2002, pp. 694-699.
13. A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd ICML*, 2005, pp. 625-632.
14. G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review*, vol. 78, no. 1, pp. 1-3, 1950.
15. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th ICML (PMLR 70)*, 2017, pp. 1321-1330.
16. L. S. Shapley, "A value for n-person games," in *Contributions to the Theory of Games II*, Princeton University Press, 1953, pp. 307-317.
17. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30*, 2017.
18. S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, pp. 56-67, 2020.
19. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
20. M. Hardt, E. Price and N. Srebro, "Equality of opportunity in supervised learning," in *Advances in Neural Information Processing Systems 29*, 2016.
21. J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy and A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, article 44, 2014.
22. S. Moro, P. Rita and P. Cortez, "Bank Marketing" [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5K306 (CC BY 4.0).
