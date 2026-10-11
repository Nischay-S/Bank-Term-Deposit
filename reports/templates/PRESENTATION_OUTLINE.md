# Presentation Outline (14 slides, about 15 minutes)

Use the screenshots you capture from the running app (see the final report, section 19).

| # | Slide | Content | Visual | Speaker notes |
|---|---|---|---|---|
| 1 | Title | BankSmart AI; names, guide, institution | logo/title | Introduce the problem in one sentence. |
| 2 | Problem and motivation | Most telemarketing calls fail; ranking customers saves effort; accuracy is misleading when positives are @@PREV@@ | class-distribution chart | "Always predict no" is @@ALWAYS_NO_ACC@@ accurate and useless. |
| 3 | Objectives | The 7 objectives, short form | list | Stress honest evaluation and explainability. |
| 4 | Dataset | UCI Bank Marketing, @@N_ROWS@@ rows, 16 inputs, CC BY 4.0, DOI 10.24432/C5K306 | schema table | Mention the supplied `test.csv` is fully contained in `train.csv`, so it was not used. |
| 5 | Two scenarios and leakage | Pre-call vs post-call; `duration` only known after the call; enforced by pipeline and test | scenario diagram | This is the most important design decision. |
| 6 | Architecture | data -> preprocessing -> training -> artifacts -> app (7 pages) | architecture diagram | Training is separate from app start-up. |
| 7 | Methodology | 60/20/20 split, CV on train, select on validation, one test evaluation, calibration, threshold | flow chart | The test set is never used for choices. |
| 8 | Imbalance and models | Eight candidates incl. class weights and SMOTE inside the pipeline | CV table | @@IMBALANCE_FINDING@@ |
| 9 | Results (pre-call) | PR-AUC @@PRE_PRAUC@@, ROC-AUC @@PRE_ROC@@, recall @@PRE_REC@@, precision @@PRE_PREC@@ with CIs | PR curve, confusion matrix | Explain the false positive / false negative trade-off and the F2 assumption. |
| 10 | Calibration and threshold | Chosen calibration: @@PRE_CAL@@; threshold @@PRE_THR@@ | reliability plot, threshold chart | A threshold encodes a cost assumption. |
| 11 | Explainable AI | SHAP top features: @@PRE_TOP_SHAP@@; association not causation | SHAP summary plot, one individual explanation | Contributions describe model behaviour. |
| 12 | Campaign analytics | Top 10/20/30% lift table (offline) | gains curve | No ROI claim; a randomised pilot is needed. |
| 13 | Honest limits | Time-ordered check: ROC-AUC @@TEMP_TEST_ROC@@ vs @@TEMP_RAND_ROC@@; no costs; no fairness audit | temporal table | The random split is optimistic; say so before you are asked. |
| 14 | Conclusion and future work | Contributions; fairness audit, drift monitoring, pilot, cost-based threshold | summary | Invite questions; demo the app if time allows. |

**Live demo order (3 minutes):** switch scenario in the sidebar -> predict one customer -> upload the demo file -> show Model Performance threshold tab -> Explainable AI individual tab.
