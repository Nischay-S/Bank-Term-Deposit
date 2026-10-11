"""Leakage-aware feature engineering and preprocessing pipeline."""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)

ENGINEERED_FEATURES = ["previously_contacted", "negative_balance"]


def _scenario_columns(scenario, exclude=()):
    """Scenario columns minus ablation exclusions (pdays/balance feed derived features, so are protected)."""
    protected = set(exclude) & {"pdays", "balance"}
    if protected:
        raise ValueError(f"Cannot exclude {sorted(protected)}: required for derived features.")
    return [c for c in config.scenario_features(scenario) if c not in set(exclude)]


class FeatureMismatchError(ValueError):
    """Raised when input data lacks columns the model requires."""


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Selects the scenario's raw columns and adds simple derived features.

    - Missing required columns -> FeatureMismatchError (never silently invented).
    - Extra columns (including a leaked 'duration' in the pre-call scenario) are dropped.
    - pdays == -1 means "never contacted before": encoded as a flag, and pdays set to 0.
    """

    exclude = ()  # class-level default keeps artifacts pickled before this option existed loadable

    def __init__(self, scenario: str = "pre_call", exclude=()):
        self.scenario = scenario
        self.exclude = exclude  # extra raw columns to drop (feature-ablation studies only)

    def fit(self, X, y=None):
        self.required_columns_ = _scenario_columns(self.scenario, self.exclude)
        return self

    def transform(self, X):
        required = _scenario_columns(self.scenario, self.exclude)
        if not isinstance(X, pd.DataFrame):
            raise FeatureMismatchError("Input must be a pandas DataFrame with named columns.")
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise FeatureMismatchError(
                f"Missing required columns for scenario '{self.scenario}': {missing}"
            )
        extra = [c for c in X.columns if c not in required]
        if extra:
            logger.warning("Ignoring columns not used by scenario '%s': %s", self.scenario, extra)
        out = X[required].copy()
        for col in config.NUMERIC_FEATURES:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out["previously_contacted"] = np.where(out["pdays"].isna(), np.nan, (out["pdays"] != -1).astype(float))
        out["pdays"] = out["pdays"].where(out["pdays"] != -1, 0)
        out["negative_balance"] = np.where(out["balance"].isna(), np.nan, (out["balance"] < 0).astype(float))
        for col in config.CATEGORICAL_FEATURES:
            if col in out.columns:
                out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
        return out


def numeric_columns(scenario: str, exclude=()) -> list:
    return [c for c in _scenario_columns(scenario, exclude) if c in config.NUMERIC_FEATURES] + ENGINEERED_FEATURES


def categorical_columns(scenario: str, exclude=()) -> list:
    return [c for c in _scenario_columns(scenario, exclude) if c in config.CATEGORICAL_FEATURES]


def build_preprocessor(scenario: str = "pre_call", scale_numeric: bool = True, exclude=()) -> Pipeline:
    """Unfitted feature engineering + encoding pipeline (no model attached)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    ct = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_columns(scenario, exclude)),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_columns(scenario, exclude)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("features", FeatureEngineer(scenario=scenario, exclude=exclude)), ("encode", ct)])
