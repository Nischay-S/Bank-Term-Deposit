# Viva Preparation: 36 Questions with Answers

Numbers in these answers come from the saved artifacts. If you retrain with different settings, rebuild with `python -m src.build_report`.

## A. Problem and data

**1. What problem does your project solve?**
It estimates the probability that a contacted bank customer subscribes to a term deposit, so that phone campaigns can prioritise customers, with explanations and an honest evaluation.

**2. Which dataset did you use and where is it from?**
UCI Bank Marketing (dataset 222), direct phone campaigns of a Portuguese bank, CC BY 4.0, DOI 10.24432/C5K306. I used the 17-column version with @@N_ROWS@@ rows, of which @@N_POS@@ (@@PREV@@) subscribed. I obtained it through a Kaggle copy and cite the UCI original.

**3. What is the class imbalance and why does it matter?**
Only about @@PREV@@ of customers subscribe. A model that always predicts "no" gets @@ALWAYS_NO_ACC@@ accuracy while finding no subscribers, so accuracy is misleading and metrics such as recall, precision and PR-AUC are needed.

**4. What was wrong with the supplied test file?**
Every one of its 4,521 rows also appears exactly in the training file, so scoring on it would test on training data. I ignored it and made my own stratified train/validation/test split.

**5. What is data leakage and where could it occur here?**
Leakage is using information that would not be available when the prediction is needed. Call `duration` is only known after the call, so using it to decide whom to call is leakage. Learning preprocessing statistics or SMOTE samples from validation/test data would also leak, so those steps are inside the training folds.

**6. How did you handle `duration`?**
I built two scenarios. The pre-call pipeline drops `duration` itself, and a test fails if it appears. The post-call scenario includes it and is shown for comparison only. They have separate models and artifact folders.

**7. Is `campaign` also leakage?**
It counts contacts in this campaign including the current one, so it is borderline. An ablation removing it changed test PR-AUC by @@ABL_CAMPAIGN@@.

**8. How did you treat "unknown" values and `pdays = -1`?**
"unknown" is kept as its own category. `pdays = -1` means never contacted before, so I created a `previously_contacted` flag and set `pdays` to 0 for those customers, instead of treating -1 as a real number of days.

## B. Method

**9. How did you split the data and why?**
Stratified random 60/20/20 (train/validation/test) with seed @@SEED@@, preserving the class ratio in each part. Cross-validation uses the training part only; validation chooses model, calibration and threshold; the test part is scored after those choices.

**10. Why separate validation and test sets?**
If the test set influenced any choice, its score would be optimistic. Validation is for decisions; test is for a final, unbiased report.

**11. How did you handle imbalance?**
I compared a plain logistic regression baseline with class-weighted and SMOTE variants (SMOTE inside the pipeline so it only touches training folds), and tree models with balanced weights. @@IMBALANCE_FINDING@@

**12. What is SMOTE and what is its limitation here?**
SMOTE creates synthetic minority samples by interpolating between a minority point and a near minority neighbour. Because my data include one-hot columns, interpolation produces fractional category values, which is not realistic.

**13. Which models did you compare and which was selected?**
Eight: logistic regression (plain, class-weighted, SMOTE), decision tree, random forest, histogram gradient boosting, and tuned forest and boosting. The pre-call selection was @@PRE_MODEL@@ (highest validation PR-AUC); the top three were @@TOP_MODELS@@.

**14. Can you claim that model is the best?**
Only tentatively. The gap between the first and third model on validation PR-AUC is @@TOP_GAP@@, compared with a cross-validation standard deviation of about @@CV_STD_SEL@@. The ordering among the top models is not decisive.

**15. How did you tune hyper-parameters?**
Randomised search with 5 candidate settings and 3-fold inner cross-validation on the training split, scoring average precision, for the forest and boosting models. It was deliberately light, which is a stated limitation.

## C. Metrics

**16. Define precision, recall and F1.**
Precision = TP/(TP+FP): of flagged customers, the share who subscribe. Recall = TP/(TP+FN): of subscribers, the share found. F1 is their harmonic mean. F2 weights recall twice as heavily as precision.

**17. Why PR-AUC and not only ROC-AUC?**
ROC-AUC uses the false-positive rate, which stays small when negatives are numerous, so it can look good even when precision is poor. PR-AUC focuses on the positive class; its random baseline equals the prevalence (@@PREV@@). I report both.

**18. What is balanced accuracy?**
The mean of recall on positives and recall on negatives; it does not reward predicting the majority class.

**19. What do your pre-call results mean in plain words?**
On the test set the model reaches ROC-AUC @@PRE_ROC@@ and PR-AUC @@PRE_PRAUC@@ against a base rate of @@PREV@@. At its chosen threshold it finds @@PRE_REC@@ of subscribers, but only @@PRE_PREC@@ of flagged customers actually subscribe.

**20. Explain false positives versus false negatives for this problem.**
A false positive is a wasted call; a false negative is a missed subscriber. At the chosen threshold the test set has @@PRE_FP@@ false positives and @@PRE_FN@@ false negatives. The F2 threshold accepts more of the former to reduce the latter, assuming a missed subscriber costs more than an extra call.

**21. How did you get confidence intervals?**
By a percentile bootstrap: resample the test set 1,000 times and recompute each metric. For pre-call PR-AUC the 95% interval is @@PRE_PRAUC_CI@@. It reflects test-sample variation only, not variation from training or from time.

## D. Calibration and thresholds

**22. What is calibration and why does it matter?**
A calibrated model's predicted 20% corresponds to about 20% of such customers subscribing. It matters when probabilities are shown to people or used in cost calculations.

**23. How did you choose the calibration method?**
I compared none, sigmoid (Platt) and isotonic calibration, each fitted with internal cross-validation on the training split, and picked the lowest Brier score on validation data. Pre-call chose @@PRE_CAL@@, post-call chose @@POST_CAL@@.

**24. How did you choose the decision threshold?**
By maximising F2 on validation probabilities (threshold @@PRE_THR@@ pre-call). That encodes an assumption about costs. With real costs, the optimal threshold for a calibrated model is c_FP/(c_FP + c_FN); the app lets the user move the threshold.

## E. Explainability

**25. What is SHAP in simple terms?**
For one prediction, each feature gets a value showing how much it pushed the model's score up or down relative to an average customer. The values add up to the prediction, which I verified numerically (maximum probability error @@PRE_ADD_ERR@@).

**26. What do SHAP values not tell you?**
They describe how the model used a feature, based on associations in historical data. They do not show that changing the feature would change the customer's behaviour.

**27. What if SHAP does not work for a model?**
The explainer falls back to a linear explainer or to occlusion (replace a feature with a typical value and see how the probability changes), and permutation importance works for any model. A test forces SHAP to fail and checks the fallback.

**28. Which features mattered most?**
Pre-call SHAP top five: @@PRE_TOP_SHAP@@. Permutation top five: @@PRE_TOP_PERM@@. They share @@PRE_TOPK_OVERLAP@@ of five.

## F. Campaign and honesty

**29. What is lift?**
Precision among the selected top customers divided by the base rate. A lift of 3 means selected customers are three times as likely to be subscribers as customers chosen at random.

**30. Can you claim the model will raise revenue?**
No. The lift tables are offline: they show how the model ranked past customers. Revenue, ROI or uplift would need cost and revenue data and a randomised experiment.

**31. What did your time-ordered check show?**
The file is ordered by date. Training on the first 60% and testing on the last 20% gave pre-call ROC-AUC @@TEMP_TEST_ROC@@, compared with @@TEMP_RAND_ROC@@ under the random split, and the subscription rate drifted from @@TEMP_PREV_TRAIN@@ to @@TEMP_PREV_TEST@@. So the random split is optimistic for forecasting a later period.

**32. Why did you still use a random split?**
The project brief instructed treating rows as randomly ordered. I kept that as the main protocol, added the temporal check as a sensitivity analysis, and disclose the difference.

## G. Engineering and ethics

**33. How do you handle bad uploads?**
The upload parser rejects wrong extensions, empty, oversized or non-UTF-8 files and duplicate or missing headers. Missing required columns raise an error that names them; near-matches are suggested but never applied. Non-numeric or out-of-range values make a row invalid; missing cells and unseen categories are scored but flagged.

**34. How is the work reproducible?**
Fixed seed (@@SEED@@), pinned requirements, saved metadata with package versions and configuration, training separate from app start-up, and a pipeline that regenerates every artifact with documented commands.

**35. What did you test?**
Data loading and validation, preprocessing (unseen categories, missing values, mismatches), the exclusion of `duration`, training wiring, prediction (ranges, thresholds, errors), SHAP additivity and fallbacks, campaign arithmetic, upload parsing, and every app page in both scenarios. I also injected bugs to confirm the tests fail when they should.

**36. What are the ethical concerns and what would you do next?**
Age and marital status are inputs, so a fairness audit (for example equality of opportunity) is needed, and customers should never be excluded solely because of a sensitive attribute. Uploaded data should be handled lawfully and people must be able to decline contact. Next: fairness audit, time-aware validation and drift monitoring, cost-based thresholds, and a randomised pilot.
