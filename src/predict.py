"""Prediction interface: artifact loading, strict input validation, single and batch prediction.

Rules enforced here (never silently "fix" customer data):
  * required columns must exist with their exact names (missing -> FeatureMismatchError);
  * extra columns are ignored and reported ('duration' in the pre-call scenario is a leakage warning);
  * non-numeric or out-of-range numeric values make that ROW invalid (no prediction is made for it);
  * missing cells and unseen categories are still predicted (the pipeline imputes / zero-encodes) but
    flagged on the row so users know the estimate rests on incomplete or unfamiliar input.
"""
import difflib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.preprocessing import FeatureMismatchError

logger = logging.getLogger(__name__)


class ArtifactsMissingError(FileNotFoundError):
    """Trained artifacts are missing; run the training pipeline."""


class InputValidationError(ValueError):
    """Raised for invalid single-customer input or an invalid threshold."""


NUMERIC_RANGES = {
    "age": (18, 100), "balance": (-100_000, 1_000_000), "day": (1, 31), "duration": (0, 10_000),
    "campaign": (1, 100), "pdays": (-1, 1_000), "previous": (0, 100),
}


# ------------------------------------------------------------------ bundle
@dataclass
class ModelBundle:
    scenario: str
    model: object          # calibrated (or raw) pipeline: raw DataFrame -> probability
    base_model: object     # underlying uncalibrated pipeline (explanations are computed on this)
    metadata: dict
    known_categories: dict = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def input_features(self) -> list:
        return list(self.metadata["input_features"])


def _known_categories(pipe, columns_hint=None) -> dict:
    steps = pipe.named_steps
    enc = steps.get("encode")
    if enc is None and "prep" in steps and hasattr(steps["prep"], "named_steps"):
        enc = steps["prep"].named_steps.get("encode")
    if enc is None:
        return {}
    for name, trans, cols in enc.transformers_:
        if name == "cat":
            onehot = trans.named_steps["onehot"]
            return {c: set(map(str, cats)) for c, cats in zip(cols, onehot.categories_)}
    return {}


def load_bundle(scenario: str, artifact_dir=None) -> ModelBundle:
    root = Path(artifact_dir) if artifact_dir else config.ARTIFACT_DIR / scenario
    config.scenario_features(scenario)  # validates scenario name
    needed = [root / "model.joblib", root / "base_model.joblib", root / "metadata.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise ArtifactsMissingError(
            f"Trained artifacts for scenario '{scenario}' not found in {root} (missing: {missing}). "
            f"Create them with:  python -m src.train --scenario {scenario}")
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("scenario") != scenario:
        raise ArtifactsMissingError(
            f"Artifacts in {root} were trained for scenario '{metadata.get('scenario')}', not '{scenario}'.")
    if list(metadata["input_features"]) != config.scenario_features(scenario):
        raise ArtifactsMissingError(
            f"Artifacts in {root} use a different feature list than the current config; retrain with "
            f"python -m src.train --scenario {scenario}")
    base = joblib.load(root / "base_model.joblib")
    return ModelBundle(scenario, joblib.load(root / "model.joblib"), base, metadata, _known_categories(base))


# ------------------------------------------------------------------ validation
@dataclass
class ValidationReport:
    ignored_columns: list
    leakage_columns_ignored: list
    n_rows: int
    n_invalid: int
    n_with_warnings: int
    unseen_categories: dict
    missing_cells: int


def validate_frame(df: pd.DataFrame, bundle: ModelBundle):
    """Validate a customer table. Returns (X_required_columns, notes, invalid_mask, report)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise InputValidationError("Input must be a non-empty table.")
    required = bundle.input_features
    missing = [c for c in required if c not in df.columns]
    if missing:
        hints = {m: difflib.get_close_matches(m, list(df.columns), n=1) for m in missing}
        hint_txt = "; ".join(f"'{m}' (did you mean '{h[0]}'?)" for m, h in hints.items() if h)
        raise FeatureMismatchError(
            f"Missing required columns for the {bundle.scenario} model: {missing}."
            + (f" Possible near-matches (NOT applied automatically): {hint_txt}." if hint_txt else "")
            + f" Required columns: {required}")
    ignored = [c for c in df.columns if c not in required]
    leak = [c for c in ignored if c in config.LEAKAGE_FEATURES[bundle.scenario]]
    if leak:
        logger.warning("Column(s) %s are not available before a call and are ignored.", leak)

    X = df[required].copy()
    notes = pd.Series([[] for _ in range(len(X))], index=X.index, dtype=object)
    invalid = pd.Series(False, index=X.index)
    missing_cells = 0
    unseen = {}

    for col in config.NUMERIC_FEATURES:
        if col not in required:
            continue
        raw = X[col]
        num = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & num.isna() & (raw.astype(str).str.strip() != "")
        lo, hi = NUMERIC_RANGES[col]
        out_of_range = num.notna() & ((num < lo) | (num > hi))
        miss = num.isna() & ~bad
        for idx in X.index[bad]:
            notes[idx].append(f"INVALID: non-numeric '{col}'")
        for idx in X.index[out_of_range]:
            notes[idx].append(f"INVALID: '{col}' outside allowed range [{lo}, {hi}]")
        for idx in X.index[miss]:
            notes[idx].append(f"'{col}' missing (training median used)")
        invalid |= bad | out_of_range
        missing_cells += int(miss.sum())
        X[col] = num

    for col in config.CATEGORICAL_FEATURES:
        if col not in required:
            continue
        vals = X[col].astype(object)
        vals = vals.where(vals.notna() & (vals.astype(str).str.strip() != ""), np.nan)
        X[col] = vals
        for idx in X.index[vals.isna()]:
            notes[idx].append(f"'{col}' missing (treated as its own 'missing' level)")
        missing_cells += int(vals.isna().sum())
        known = bundle.known_categories.get(col)
        if known:
            is_new = vals.notna() & ~vals.astype(str).isin(known)
            for idx in X.index[is_new]:
                notes[idx].append(f"unseen category {col}='{vals[idx]}' (no information from this field)")
            if is_new.any():
                unseen[col] = sorted(set(map(str, vals[is_new])))

    note_txt = notes.map(lambda n: "; ".join(n))
    report = ValidationReport(ignored, leak, len(X), int(invalid.sum()),
                              int(((note_txt != "") & ~invalid).sum()), unseen, missing_cells)
    return X, note_txt, invalid, report


# ------------------------------------------------------------------ prediction
def _check_threshold(threshold, bundle):
    t = bundle.threshold if threshold is None else threshold
    if not isinstance(t, (int, float, np.floating)) or not (0.0 <= float(t) <= 1.0):
        raise InputValidationError(f"Threshold must be a number between 0 and 1, got {threshold!r}.")
    return float(t)


def predict_frame(df: pd.DataFrame, bundle: ModelBundle, threshold=None):
    """Predict for every valid row. Returns (results, report); invalid rows get no prediction."""
    thr = _check_threshold(threshold, bundle)
    X, notes, invalid, report = validate_frame(df, bundle)
    out = pd.DataFrame(index=X.index)
    out["probability"] = np.nan
    ok = ~invalid
    if ok.any():
        out.loc[ok, "probability"] = bundle.model.predict_proba(X.loc[ok])[:, 1]
    out["predicted_class"] = pd.array(
        np.where(out["probability"].isna(), pd.NA, (out["probability"] >= thr).astype(float)), dtype="Int64")
    out["predicted_label"] = out["predicted_class"].map(
        {1: config.POSITIVE_LABEL, 0: config.NEGATIVE_LABEL}).astype(object)
    out["status"] = np.where(invalid, "invalid", np.where(notes != "", "warning", "ok"))
    out["data_quality_note"] = notes
    out.attrs["threshold"] = thr
    return out, report


def predict_single(values: dict, bundle: ModelBundle, threshold=None) -> dict:
    """Predict one customer from a dict of raw attributes; raises on invalid input."""
    res, _ = predict_frame(pd.DataFrame([values]), bundle, threshold)
    row = res.iloc[0]
    if row["status"] == "invalid":
        raise InputValidationError(row["data_quality_note"])
    return {"probability": float(row["probability"]), "predicted_class": int(row["predicted_class"]),
            "predicted_label": row["predicted_label"], "threshold": res.attrs["threshold"],
            "notes": row["data_quality_note"]}
