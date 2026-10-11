"""Explainable AI: SHAP (primary), permutation importance + occlusion (documented fallbacks),
logistic-regression reference coefficients and plain-language explanations.

Explanations are computed on the UNCALIBRATED base model (artifacts/<scenario>/base_model.joblib)
using the exact preprocessing steps stored inside it. Calibration is a near-monotone rescaling of the
same score, so the ranking of contributions is unchanged. They describe how the MODEL uses inputs
(statistical association in historical data) - they are not causal claims.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src import data_loader as dl
from src.predict import ModelBundle, load_bundle
from src.preprocessing import build_preprocessor, categorical_columns

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "age": "age", "job": "occupation", "marital": "marital status", "education": "education level",
    "default": "credit in default", "balance": "account balance", "housing": "housing loan",
    "loan": "personal loan", "contact": "contact channel", "day": "day of month contacted",
    "month": "month contacted", "duration": "last call duration (s)", "campaign": "contacts this campaign",
    "pdays": "days since previous campaign contact", "previous": "contacts before this campaign",
    "poutcome": "previous campaign outcome", "previously_contacted": "previously contacted (flag)",
    "negative_balance": "negative balance (flag)",
}
ASSOCIATION_NOTE = ("These statements describe how the model used the inputs, reflecting associations in "
                    "historical data. They are not causes: changing a customer's attributes would not "
                    "necessarily change their behaviour.")


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


def _split_pipeline(base):
    """Return (preprocessing_pipeline, final_estimator) for nested or flattened (SMOTE) pipelines."""
    if "prep" in base.named_steps:
        prep = base.named_steps["prep"]
    else:
        prep = Pipeline([(n, s) for n, s in base.steps if n in ("features", "encode")])
    return prep, base.steps[-1][1]


def feature_group(column, scenario):
    for c in categorical_columns(scenario):
        if column.startswith(c + "_"):
            return c
    return column


# ------------------------------------------------------------------ explainer
class ModelExplainer:
    """Wraps SHAP with graceful fallbacks. `method` is one of: shap_tree, shap_linear, occlusion."""

    def __init__(self, bundle: ModelBundle, background: pd.DataFrame, method: str = "auto"):
        self.bundle, self.scenario = bundle, bundle.scenario
        self.features = bundle.input_features
        self.prep, self.clf = _split_pipeline(bundle.base_model)
        self.background = background[self.features].reset_index(drop=True)
        self.method, self.output_space, self._shap = "occlusion", "probability (occlusion)", None
        if method == "auto":
            self._init_shap()
        self._reference = self._reference_row()

    def _init_shap(self):
        try:
            import shap
            try:
                self._shap = shap.TreeExplainer(self.clf)
                self.method = "shap_tree"
            except Exception:
                if not hasattr(self.clf, "coef_"):
                    raise
                self._shap = shap.LinearExplainer(self.clf, self.prep.transform(self.background.head(200)))
                self.method = "shap_linear"
            self.output_space = "probability" if "Forest" in type(self.clf).__name__ else "log-odds"
        except Exception as exc:  # documented fallback instead of crashing
            logger.warning("SHAP unavailable for %s (%s); using occlusion fallback.", type(self.clf).__name__, exc)
            self._shap, self.method, self.output_space = None, "occlusion", "probability (occlusion)"

    def _reference_row(self):
        ref = {}
        for c in self.features:
            col = self.background[c]
            ref[c] = float(col.median()) if c in config.NUMERIC_FEATURES else col.mode().iloc[0]
        return ref

    # ---- SHAP values on transformed columns
    def shap_matrix(self, X: pd.DataFrame):
        if self._shap is None:
            raise RuntimeError("SHAP not available; use occlusion explanations.")
        Xt = self.prep.transform(X[self.features])
        sv = self._shap.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        base = np.atleast_1d(np.asarray(self._shap.expected_value, dtype=float))[-1]
        return sv, float(base), Xt

    # ---- local explanation
    def explain_row(self, row: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions for ONE customer (DataFrame with one row of raw features)."""
        if len(row) != 1:
            raise ValueError("explain_row expects exactly one row.")
        if self.method == "occlusion":
            return self._occlusion(row)
        sv, base, Xt = self.shap_matrix(row)
        groups = {}
        for j, col in enumerate(Xt.columns):
            groups.setdefault(feature_group(col, self.scenario), []).append(j)
        rows = []
        for g, idx in groups.items():
            if g in row.columns:
                val = row.iloc[0][g]
            else:
                val = Xt.iloc[0][g]
            rows.append({"feature": g, "value": val, "contribution": float(sv[0, idx].sum())})
        out = pd.DataFrame(rows)
        out["abs_contribution"] = out["contribution"].abs()
        out = out.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
        out.attrs.update(base_value=base, method=self.method, output_space=self.output_space)
        return out

    def _occlusion(self, row):
        p_full = float(self.bundle.base_model.predict_proba(row[self.features])[:, 1][0])
        rows = []
        for c in self.features:
            mod = row[self.features].copy()
            mod[c] = pd.Series([self._reference[c]], index=mod.index).astype(mod[c].dtype, errors="ignore")
            p_mod = float(self.bundle.base_model.predict_proba(mod)[:, 1][0])
            rows.append({"feature": c, "value": row.iloc[0][c], "contribution": p_full - p_mod})
        out = pd.DataFrame(rows)
        out["abs_contribution"] = out["contribution"].abs()
        out = out.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
        out.attrs.update(base_value=None, method="occlusion", output_space=self.output_space)
        return out


def describe_explanation(expl: pd.DataFrame, probability: float, threshold: float, top_k: int = 3) -> list:
    """Plain-language explanation of one prediction (association, not causation)."""
    total = expl["abs_contribution"].sum() or 1.0

    def strength(c):
        share = abs(c) / total
        return "strongly" if share >= 0.25 else "moderately" if share >= 0.10 else "slightly"

    def fmt(f, v):
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return f"{v:g}"
        return str(v)

    side = "above" if probability >= threshold else "below"
    lines = [f"The model estimates a {probability:.1%} probability of subscription, {side} the decision "
             f"threshold of {threshold:.1%}."]
    for _, r in expl[expl.contribution > 0].head(top_k).iterrows():
        lines.append(f"{label(r.feature).capitalize()} = {fmt(r.feature, r.value)} {strength(r.contribution)} "
                     f"raised the model's estimate.")
    for _, r in expl[expl.contribution < 0].head(top_k).iterrows():
        lines.append(f"{label(r.feature).capitalize()} = {fmt(r.feature, r.value)} {strength(r.contribution)} "
                     f"lowered the model's estimate.")
    lines.append(ASSOCIATION_NOTE)
    return lines


# ------------------------------------------------------------------ global explanations
def global_importance(explainer: ModelExplainer, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature (one-hot columns summed back to their source feature)."""
    sv, _, Xt = explainer.shap_matrix(X_sample)
    df = pd.DataFrame(np.abs(sv), columns=Xt.columns)
    # sum signed values per group per row first, then take |.| (matches individual explanations)
    signed = pd.DataFrame(sv, columns=Xt.columns)
    grouped = signed.T.groupby(lambda c: feature_group(c, explainer.scenario)).sum().T
    out = grouped.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    out.columns = ["feature", "mean_abs_shap"]
    out["label"] = out["feature"].map(label)
    return out


def permutation_table(bundle: ModelBundle, X: pd.DataFrame, y, n_repeats=5, n_max=4000, seed=config.RANDOM_SEED):
    """Model-agnostic fallback: drop in PR-AUC when each raw feature is shuffled (validation data)."""
    idx = X.sample(min(n_max, len(X)), random_state=seed).index
    Xs, ys = X.loc[idx, bundle.input_features], y.loc[idx]
    res = permutation_importance(bundle.base_model, Xs, ys, scoring="average_precision",
                                 n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": bundle.input_features, "pr_auc_drop_mean": res.importances_mean,
                          "pr_auc_drop_std": res.importances_std})
            .sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True))


def logistic_reference(scenario, Xtr, ytr) -> pd.DataFrame:
    """Interpretable reference: class-weighted logistic regression coefficients.
    Numeric inputs are standardised (coefficient = effect per 1 SD); one-hot levels are relative to
    the other levels of the same feature. Coefficients are associations, not causal effects."""
    pipe = Pipeline([("prep", build_preprocessor(scenario)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED))])
    pipe.fit(Xtr[config.scenario_features(scenario)], ytr)
    names = pipe.named_steps["prep"].named_steps["encode"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    return (pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
            .assign(abs_coef=lambda d: d.coefficient.abs()).sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef").reset_index(drop=True))


def shap_summary_figure(npz_path, max_display=15):
    """Beeswarm summary plot from a saved shap_sample.npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    d = np.load(npz_path, allow_pickle=True)
    expl = shap.Explanation(values=d["values"], base_values=np.full(len(d["values"]), float(d["base_value"])),
                            data=d["data"], feature_names=list(d["feature_names"]))
    plt.figure(figsize=(8, 6))
    shap.summary_plot(expl.values, expl.data, feature_names=expl.feature_names, max_display=max_display, show=False)
    plt.tight_layout()
    return plt.gcf()


# ------------------------------------------------------------------ artifact generation
def build_explanation_artifacts(scenario, n_shap=2000):
    bundle = load_bundle(scenario)
    X, y = dl.split_features_target(dl.load_raw())
    s = dl.make_splits(X, y)
    out = config.ARTIFACT_DIR / scenario / "explain"
    out.mkdir(parents=True, exist_ok=True)
    feats = bundle.input_features
    # explanations use VALIDATION rows so the test split stays untouched
    sample = s.X_val[feats].sample(min(n_shap, len(s.X_val)), random_state=config.RANDOM_SEED)
    expl = ModelExplainer(bundle, background=s.X_train[feats].sample(500, random_state=config.RANDOM_SEED))
    info = {"scenario": scenario, "method": expl.method, "output_space": expl.output_space,
            "estimator": type(expl.clf).__name__, "n_explained": int(len(sample))}
    if expl.method != "occlusion":
        gi = global_importance(expl, sample)
        gi.to_csv(out / "global_shap.csv", index=False)
        sv, base, Xt = expl.shap_matrix(sample)
        np.savez_compressed(out / "shap_sample.npz", values=sv, data=Xt.values,
                            feature_names=np.array(Xt.columns), base_value=base)
        # additivity check: base + sum(SHAP) should reproduce the model's raw score
        if expl.output_space == "log-odds":
            p_shap = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
            p_model = bundle.base_model.predict_proba(sample)[:, 1]
            info["max_abs_additivity_error"] = float(np.max(np.abs(p_shap - p_model)))
    permutation_table(bundle, s.X_val, s.y_val).to_csv(out / "permutation_importance.csv", index=False)
    logistic_reference(scenario, s.X_train, s.y_train).to_csv(out / "logreg_coefficients.csv", index=False)
    (out / "explain_info.json").write_text(json.dumps(info, indent=2))
    logger.info("[%s] explanation artifacts written: %s", scenario, info)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(config.SCENARIOS) + ["all"], default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for sc in (config.SCENARIOS if args.scenario == "all" else [args.scenario]):
        build_explanation_artifacts(sc)


if __name__ == "__main__":
    main()
