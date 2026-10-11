# Presentation Outline (14 slides, about 15 minutes)

Use the screenshots you capture from the running app (see the final report, section 19).

| # | Slide | Content | Visual | Speaker notes |
|---|---|---|---|---|
| 1 | Title | BankSmart AI; names, guide, institution | logo/title | Introduce the problem in one sentence. |
| 2 | Problem and motivation | Most telemarketing calls fail; ranking customers saves effort; accuracy is misleading when positives are 11.70% | class-distribution chart | "Always predict no" is 88.3% accurate and useless. |
| 3 | Objectives | The 7 objectives, short form | list | Stress honest evaluation and explainability. |
| 4 | Dataset | UCI Bank Marketing, 45,211 rows, 16 inputs, CC BY 4.0, DOI 10.24432/C5K306 | schema table | Mention the supplied `test.csv` is fully contained in `train.csv`, so it was not used. |
| 5 | Two scenarios and leakage | Pre-call vs post-call; `duration` only known after the call; enforced by pipeline and test | scenario diagram | This is the most important design decision. |
| 6 | Architecture | data -> preprocessing -> training -> artifacts -> app (7 pages) | architecture diagram | Training is separate from app start-up. |
| 7 | Methodology | 60/20/20 split, CV on train, select on validation, one test evaluation, calibration, threshold | flow chart | The test set is never used for choices. |
| 8 | Imbalance and models | Eight candidates incl. class weights and SMOTE inside the pipeline | CV table | On validation data the three logistic-regression variants reached PR-AUC 0.414 (plain), 0.409 (class-weighted) and 0.410 (SMOTE), a spread of 0.004, so imbalance handling changed ranking quality very little. At the default 0.5 cut-off, cross-validated recall was 0.165 for the plain model versus 0.622 (class-weighted) and 0.620 (SMOTE): imbalance handling mainly moves where the default cut-off falls, which a tuned threshold can also do. |
| 9 | Results (pre-call) | PR-AUC 0.455, ROC-AUC 0.801, recall 0.648, precision 0.325 with CIs | PR curve, confusion matrix | Explain the false positive / false negative trade-off and the F2 assumption. |
| 10 | Calibration and threshold | Chosen calibration: none; threshold 0.113 | reliability plot, threshold chart | A threshold encodes a cost assumption. |
| 11 | Explainable AI | SHAP top features: contact channel, month contacted, previous campaign outcome, account balance, housing loan; association not causation | SHAP summary plot, one individual explanation | Contributions describe model behaviour. |
| 12 | Campaign analytics | Top 10/20/30% lift table (offline) | gains curve | No ROI claim; a randomised pilot is needed. |
| 13 | Honest limits | Time-ordered check: ROC-AUC 0.631 vs 0.801; no costs; no fairness audit | temporal table | The random split is optimistic; say so before you are asked. |
| 14 | Conclusion and future work | Contributions; fairness audit, drift monitoring, pilot, cost-based threshold | summary | Invite questions; demo the app if time allows. |

**Live demo order (3 minutes):** switch scenario in the sidebar -> predict one customer -> upload the demo file -> show Model Performance threshold tab -> Explainable AI individual tab.
