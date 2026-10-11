import numpy as np
import pandas as pd
import pytest

import config
from src import data_loader as dl
from src.preprocessing import FeatureMismatchError, build_preprocessor


def _X(df):
    return dl.split_features_target(df)[0]


def test_pre_call_excludes_duration(synthetic_df):
    X = _X(synthetic_df)
    out = build_preprocessor("pre_call").fit_transform(X)
    assert not any("duration" in c for c in out.columns)
    assert "duration" not in config.scenario_features("pre_call")


def test_post_call_includes_duration(synthetic_df):
    out = build_preprocessor("post_call").fit_transform(_X(synthetic_df))
    assert "duration" in out.columns


def test_scenarios_have_distinct_feature_sets():
    assert set(config.scenario_features("post_call")) - set(config.scenario_features("pre_call")) == {"duration"}


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        config.scenario_features("whenever")


def test_output_has_no_nans_and_is_numeric(synthetic_df):
    out = build_preprocessor().fit_transform(_X(synthetic_df))
    assert not out.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_unseen_category_does_not_crash(synthetic_df):
    X = _X(synthetic_df)
    pipe = build_preprocessor().fit(X)
    new = X.head(3).copy()
    new["job"] = "astronaut"
    out = pipe.transform(new)
    job_cols = [c for c in out.columns if c.startswith("job_")]
    assert (out[job_cols].sum(axis=1) == 0).all()  # all-zero one-hot, not an error


def test_missing_values_are_imputed_not_fatal(synthetic_df):
    X = _X(synthetic_df)
    pipe = build_preprocessor().fit(X)
    new = X.head(4).copy()
    new.loc[new.index[0], "balance"] = np.nan
    new.loc[new.index[1], "job"] = np.nan
    new.loc[new.index[2], "pdays"] = np.nan
    out = pipe.transform(new)
    assert not out.isna().any().any()
    assert out.shape[0] == 4


def test_missing_required_column_raises(synthetic_df):
    X = _X(synthetic_df)
    pipe = build_preprocessor().fit(X)
    with pytest.raises(FeatureMismatchError, match="balance"):
        pipe.transform(X.drop(columns=["balance"]))


def test_extra_columns_ignored_and_duration_dropped_pre_call(synthetic_df):
    X = _X(synthetic_df)
    pipe = build_preprocessor("pre_call").fit(X)
    X2 = X.copy(); X2["customer_id"] = 1
    out = pipe.transform(X2)
    assert not any(c in out.columns for c in ("duration", "customer_id"))


def test_pdays_minus_one_becomes_flag_not_value(synthetic_df):
    X = _X(synthetic_df)
    fe = build_preprocessor().named_steps["features"].fit(X)
    out = fe.transform(X)
    never = X["pdays"] == -1
    assert (out.loc[never, "previously_contacted"] == 0).all()
    assert (out.loc[~never, "previously_contacted"] == 1).all()
    assert (out["pdays"] >= 0).all()


def test_real_data_pipeline_fits(real_data_path):
    X, y = dl.split_features_target(dl.load_raw(real_data_path))
    s = dl.make_splits(X, y)
    pipe = build_preprocessor("pre_call").fit(s.X_train)
    out = pipe.transform(s.X_test)
    assert out.shape[0] == len(s.X_test) and not out.isna().any().any()
    assert not any("duration" in c for c in out.columns)
