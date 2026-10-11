"""Fill the report templates with numbers read from the saved artifacts.

    python -m src.build_report

Templates live in reports/templates/*.md and contain @@TOKENS@@. Every figure in the generated documents is
taken from artifacts/ (metadata.json and CSVs), so nothing is typed by hand. Build FAILS if a token is unknown
or if a template references data that does not exist.
"""
import json
import logging
import re

import pandas as pd

import config
from src.explain import FEATURE_LABELS

logger = logging.getLogger(__name__)
TEMPLATE_DIR = config.REPORTS_DIR / "templates"
DOCS = ["FINAL_REPORT", "SYNOPSIS", "PRESENTATION_OUTLINE", "VIVA_QUESTIONS", "SOURCE_GUIDE"]
PREFIX = {"pre_call": "PRE", "post_call": "POST"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame, floats=3) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{floats}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _read(sc, name, **kw):
    return pd.read_csv(config.ARTIFACT_DIR / sc / name, **kw)


def make_tokens() -> dict:
    T = {}
    summary = json.loads((config.ARTIFACT_DIR / "dataset_summary.json").read_text())
    T.update(N_ROWS=f"{summary['n_rows']:,}", N_POS=f"{summary['n_positive']:,}",
             N_NEG=f"{summary['n_rows'] - summary['n_positive']:,}", PREV=pct(summary["prevalence"], 2),
             ALWAYS_NO_ACC=pct(1 - summary["prevalence"], 1),
             N_TRAIN=f"{summary['split_sizes']['train']:,}", N_VAL=f"{summary['split_sizes']['validation']:,}",
             N_TEST=f"{summary['split_sizes']['test']:,}",
             PDAYS_NEVER=pct(summary["never_previously_contacted_share"]),
             UNK_POUTCOME=pct(summary["unknown_share"]["poutcome"]), UNK_CONTACT=pct(summary["unknown_share"]["contact"]),
             UNK_EDU=pct(summary["unknown_share"]["education"]))
    metas = {}
    for sc, P in PREFIX.items():
        m = json.loads((config.ARTIFACT_DIR / sc / "metadata.json").read_text())
        metas[sc] = m
        t, v, ci = m["metrics_test"], m["metrics_validation"], m["test_bootstrap_ci_95"]
        T.update({
            f"{P}_MODEL": m["selected_model"].replace("_", " "), f"{P}_CAL": m["calibration_method"],
            f"{P}_THR": f"{m['threshold']:.3f}",
            f"{P}_PRAUC": f"{t['pr_auc']:.3f}", f"{P}_PRAUC_CI": f"{ci['pr_auc']['low']:.3f} to {ci['pr_auc']['high']:.3f}",
            f"{P}_ROC": f"{t['roc_auc']:.3f}", f"{P}_ROC_CI": f"{ci['roc_auc']['low']:.3f} to {ci['roc_auc']['high']:.3f}",
            f"{P}_REC": f"{t['recall']:.3f}", f"{P}_REC_CI": f"{ci['recall']['low']:.3f} to {ci['recall']['high']:.3f}",
            f"{P}_PREC": f"{t['precision']:.3f}", f"{P}_PREC_CI": f"{ci['precision']['low']:.3f} to {ci['precision']['high']:.3f}",
            f"{P}_F1": f"{t['f1']:.3f}", f"{P}_BAL": f"{t['balanced_accuracy']:.3f}", f"{P}_ACC": f"{t['accuracy']:.3f}",
            f"{P}_BRIER": f"{t['brier']:.4f}", f"{P}_ECE": f"{t['ece']:.4f}",
            f"{P}_TP": f"{t['tp']:,}", f"{P}_FP": f"{t['fp']:,}", f"{P}_FN": f"{t['fn']:,}", f"{P}_TN": f"{t['tn']:,}",
            f"{P}_NFLAG": f"{t['n_predicted_positive']:,}",
            f"{P}_VAL_PRAUC": f"{v['pr_auc']:.3f}", f"{P}_VAL_ROC": f"{v['roc_auc']:.3f}",
            f"{P}_TRAIN_SECONDS": f"{m['train_seconds']:.0f}",
        })
        cmp_ = _read(sc, "model_comparison.csv")
        cols = ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "threshold"]
        for split, tag in (("val", "VAL"), ("test", "TEST")):
            d = cmp_[cmp_.split == split][cols].sort_values("pr_auc", ascending=False).copy()
            d["model"] = d["model"].str.replace("_", " ")
            T[f"TABLE_COMP_{tag}_{P}"] = md_table(d)
        cv = _read(sc, "cv_results.csv")
        d = pd.DataFrame({"model": cv["model"].str.replace("_", " ")})
        for k, name in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall", "recall at 0.5"), ("precision", "precision at 0.5")):
            d[name] = [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv[f"{k}_mean"], cv[f"{k}_std"])]
        T[f"TABLE_CV_{P}"] = md_table(d)
        T[f"TABLE_CAL_{P}"] = md_table(_read(sc, "calibration_comparison.csv"), floats=4)
        camp = _read(sc, "campaign_table_test.csv")
        c = pd.DataFrame({"outreach": [pct(f, 0) for f in camp.outreach_fraction],
                          "customers selected": camp.customers_selected, "subscribers found": camp.subscribers_found,
                          "precision": camp.precision, "recall": camp.recall, "lift": camp.lift,
                          "found by random selection": camp.random_expected_subscribers})
        T[f"TABLE_CAMPAIGN_{P}"] = md_table(c)
        rep = m["classification_report_test"]
        T[f"TABLE_CLASSREP_{P}"] = md_table(pd.DataFrame([
            {"class": k.replace("_", " "), "precision": rep[k]["precision"], "recall": rep[k]["recall"],
             "f1-score": rep[k]["f1-score"], "support": int(rep[k]["support"])} for k in ("not_subscribed", "subscribed")]))
        ex = config.ARTIFACT_DIR / sc / "explain"
        g = pd.read_csv(ex / "global_shap.csv").head(8)
        T[f"TABLE_SHAP_{P}"] = md_table(pd.DataFrame({"feature": g["label"], "mean |SHAP| (log-odds)": g["mean_abs_shap"]}))
        p = pd.read_csv(ex / "permutation_importance.csv").head(8)
        T[f"TABLE_PERM_{P}"] = md_table(pd.DataFrame({"feature": p["feature"], "PR-AUC drop": p["pr_auc_drop_mean"],
                                                       "std": p["pr_auc_drop_std"]}))
        lr = pd.read_csv(ex / "logreg_coefficients.csv")
        T[f"TABLE_LR_POS_{P}"] = md_table(lr.sort_values("coefficient", ascending=False).head(6).round(3))
        T[f"TABLE_LR_NEG_{P}"] = md_table(lr.sort_values("coefficient").head(6).round(3))
        info = json.loads((ex / "explain_info.json").read_text())
        T[f"{P}_SHAP_METHOD"] = info["method"].replace("_", " ")
        T[f"{P}_ADD_ERR"] = f"{info.get('max_abs_additivity_error', float('nan')):.1e}"
        T[f"{P}_TOP_SHAP"] = ", ".join(g["label"].head(5))
        T[f"{P}_TOP_PERM"] = ", ".join(FEATURE_LABELS.get(f, f) for f in p["feature"].head(5))
        T[f"{P}_TUNE"] = "; ".join(f"{k.replace('_', ' ')}: inner-CV PR-AUC {v['inner_cv_pr_auc']:.3f}" for k, v in m["tuning"].items())

    pre, post = metas["pre_call"], metas["post_call"]
    T["ENV_PY"], T["ENV_SK"] = pre["environment"]["python"], pre["environment"]["scikit-learn"]
    T["ENV_NP"], T["ENV_PD"] = pre["environment"]["numpy"], pre["environment"]["pandas"]
    T["ENV_IMB"], T["ENV_SHAP"] = pre["environment"]["imbalanced-learn"], pre["environment"]["shap"]
    T["SEED"] = str(pre["training_config"]["seed"])
    T["CV_FOLDS"] = str(pre["training_config"]["cv_folds"])
    T["DUR_GAIN_PR"] = f"{post['metrics_test']['pr_auc'] - pre['metrics_test']['pr_auc']:+.3f}"
    T["DUR_GAIN_ROC"] = f"{post['metrics_test']['roc_auc'] - pre['metrics_test']['roc_auc']:+.3f}"
    T["TABLE_SCENARIOS"] = md_table(pd.DataFrame([
        {"scenario": "pre-call", "PR-AUC": pre["metrics_test"]["pr_auc"], "ROC-AUC": pre["metrics_test"]["roc_auc"],
         "recall": pre["metrics_test"]["recall"], "precision": pre["metrics_test"]["precision"]},
        {"scenario": "post-call", "PR-AUC": post["metrics_test"]["pr_auc"], "ROC-AUC": post["metrics_test"]["roc_auc"],
         "recall": post["metrics_test"]["recall"], "precision": post["metrics_test"]["precision"]}]))

    # data-driven findings (sentences are generated from the numbers, not typed)
    cmp_ = _read("pre_call", "model_comparison.csv")
    val = cmp_[cmp_.split == "val"].set_index("model")
    test = cmp_[cmp_.split == "test"].set_index("model")
    lr_names = ["logreg_baseline", "logreg_balanced", "logreg_smote"]
    a, b, c = (val.loc[n, "pr_auc"] for n in lr_names)
    spread = max(a, b, c) - min(a, b, c)
    cv = _read("pre_call", "cv_results.csv").set_index("model")
    T["IMBALANCE_FINDING"] = (
        f"On validation data the three logistic-regression variants reached PR-AUC {a:.3f} (plain), {b:.3f} "
        f"(class-weighted) and {c:.3f} (SMOTE), a spread of {spread:.3f}, so imbalance handling "
        f"{'changed ranking quality very little' if spread < 0.01 else 'changed ranking quality noticeably'}. "
        f"At the default 0.5 cut-off, cross-validated recall was {cv.loc['logreg_baseline', 'recall_mean']:.3f} for the plain model "
        f"versus {cv.loc['logreg_balanced', 'recall_mean']:.3f} (class-weighted) and {cv.loc['logreg_smote', 'recall_mean']:.3f} (SMOTE): "
        f"imbalance handling mainly moves where the default cut-off falls, which a tuned threshold can also do.")
    top = val["pr_auc"].sort_values(ascending=False)
    T["TOP_MODELS"] = ", ".join(f"{n.replace('_', ' ')} ({v:.3f})" for n, v in top.head(3).items())
    T["TOP_GAP"] = f"{top.iloc[0] - top.iloc[2]:.3f}"
    sel = pre["selected_model"]
    T["CV_STD_SEL"] = f"{cv.loc[sel, 'pr_auc_std']:.3f}"
    ab = pd.read_csv(config.ARTIFACT_DIR / "pre_call" / "ablation.csv")
    A = ab.copy()
    A["Δ test PR-AUC"] = A["delta_test_pr_auc"]
    A["95% CI of Δ"] = [("reference" if i == 0 else f"{lo:+.3f} to {hi:+.3f}") for i, (lo, hi) in enumerate(zip(A.delta_ci_low, A.delta_ci_high))]
    T["TABLE_ABLATION"] = md_table(A[["variant", "removed", "test_pr_auc", "test_roc_auc", "Δ test PR-AUC", "95% CI of Δ"]])
    rows = {r.variant: r for r in ab.itertuples()}
    cmp_row = rows["without campaign"]
    T["ABL_CAMPAIGN"] = (f"{cmp_row.delta_test_pr_auc:+.3f} (95% CI {cmp_row.delta_ci_low:+.3f} to {cmp_row.delta_ci_high:+.3f})"
                         f", which {'includes' if cmp_row.delta_ci_low <= 0 <= cmp_row.delta_ci_high else 'excludes'} zero")
    dm = rows["without day, month"]
    T["ABL_DAYMONTH"] = f"{dm.delta_test_pr_auc:+.3f} (95% CI {dm.delta_ci_low:+.3f} to {dm.delta_ci_high:+.3f})"
    sens = rows["without age, marital (sensitive attributes)"]
    T["ABL_SENS"] = f"{sens.delta_test_pr_auc:+.3f} (95% CI {sens.delta_ci_low:+.3f} to {sens.delta_ci_high:+.3f})"

    T["ACC_REL"] = "below" if pre["metrics_test"]["accuracy"] < 1 - summary["prevalence"] else "above"
    cr = cmp_row
    T["ABL_CAMPAIGN_READ"] = ("so the borderline nature of that feature does not materially affect the result"
                              if cr.delta_ci_low <= 0 <= cr.delta_ci_high else
                              "so that borderline feature does carry measurable signal and its use should be reviewed")
    T["ABL_DM_READ"] = ("is a clear loss, showing reliance on contact timing" if dm.delta_ci_high < 0 else
                        "is not clearly different from zero")
    sh = set(pd.read_csv(config.ARTIFACT_DIR / "pre_call" / "explain" / "global_shap.csv")["feature"].head(5))
    pm = set(pd.read_csv(config.ARTIFACT_DIR / "pre_call" / "explain" / "permutation_importance.csv")["feature"].head(5))
    T["PRE_TOPK_OVERLAP"] = str(len(sh & pm))
    tmp = json.loads((config.ARTIFACT_DIR / "pre_call" / "temporal_check.json").read_text())
    tt = tmp["temporal"]
    T["TEMP_TEST_ROC"], T["TEMP_VAL_ROC"] = f"{tt['test_roc_auc']:.3f}", f"{tt['val_roc_auc']:.3f}"
    T["TEMP_TEST_PR"] = f"{tt['test_pr_auc']:.3f}"
    T["TEMP_REC"], T["TEMP_PREC"] = f"{tt['test_recall']:.3f}", f"{tt['test_precision']:.3f}"
    T["TEMP_PREV_TRAIN"], T["TEMP_PREV_VAL"], T["TEMP_PREV_TEST"] = (pct(tmp["prevalence"][k]) for k in ("train", "val", "test"))
    T["TEMP_MONTH_RUNS"] = str(tmp["n_month_runs"])
    T["TEMP_RUNS_HEAD"] = ", ".join(m for m, _ in tmp["month_runs_head"])
    T["TEMP_RAND_ROC"] = f"{tmp['random_split_reference']['test_roc_auc']:.3f}"
    T["TEMP_TRAIN_MONTHS"] = ", ".join(tmp["months_in_train_block"])
    T["TABLE_TEMPORAL"] = md_table(pd.DataFrame([
        {"evaluation": "Random stratified split (main)", "test ROC-AUC": tmp["random_split_reference"]["test_roc_auc"],
         "test subscription rate": pre["metrics_test"]["prevalence"]},
        {"evaluation": "Time-ordered (first 60% train, last 20% test)", "test ROC-AUC": tt["test_roc_auc"],
         "test subscription rate": tmp["prevalence"]["test"]}]))
    # dataset tables
    rates = summary["rates_by_category"]
    mo = pd.DataFrame(rates["month"])
    T["TABLE_MONTH_RATES"] = md_table(pd.DataFrame({"month": mo["category"], "contacts": mo["n"], "subscription rate": mo["rate"].map(lambda x: pct(x))}))
    po = pd.DataFrame(rates["poutcome"])
    T["TABLE_POUTCOME_RATES"] = md_table(pd.DataFrame({"previous outcome": po["category"], "contacts": po["n"], "subscription rate": po["rate"].map(lambda x: pct(x))}))
    return T


def build():
    tokens = make_tokens()
    out = []
    for name in DOCS:
        src = TEMPLATE_DIR / f"{name}.md"
        if not src.exists():
            logger.warning("template missing: %s", src)
            continue
        text = src.read_text()
        for k in sorted(tokens, key=len, reverse=True):
            text = text.replace(f"@@{k}@@", tokens[k])
        left = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", text)))
        if left:
            raise KeyError(f"{name}: unknown tokens {left}")
        (config.REPORTS_DIR / f"{name}.md").write_text(text)
        out.append(name)
    logger.info("Built: %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
