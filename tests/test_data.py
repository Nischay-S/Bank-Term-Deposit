import pandas as pd
import pytest

import config
from src import data_loader as dl


def test_load_synthetic_csv(synthetic_csv):
    df = dl.load_raw(synthetic_csv)
    assert set(config.ALL_FEATURES + [config.TARGET]) <= set(df.columns)


def test_semicolon_separator_detected(tmp_path, synthetic_df):
    p = tmp_path / "semi.csv"
    synthetic_df.to_csv(p, index=False, sep=";")
    assert dl.load_raw(p).shape == synthetic_df.shape


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        dl.load_raw(tmp_path / "nope.csv")


def test_missing_column_rejected(synthetic_df):
    with pytest.raises(dl.DataValidationError, match="Missing required columns"):
        dl.validate_schema(synthetic_df.drop(columns=["balance"]))


def test_non_numeric_column_rejected(synthetic_df):
    synthetic_df["age"] = "old"
    with pytest.raises(dl.DataValidationError, match="numeric"):
        dl.validate_schema(synthetic_df)


def test_unexpected_category_rejected(synthetic_df):
    synthetic_df.loc[0, "contact"] = "carrier-pigeon"
    with pytest.raises(dl.DataValidationError, match="unexpected values"):
        dl.validate_schema(synthetic_df)


def test_target_must_be_yes_no(synthetic_df):
    synthetic_df.loc[0, "y"] = "maybe"
    with pytest.raises(dl.DataValidationError, match="invalid values"):
        dl.validate_target(synthetic_df["y"])


def test_target_needs_both_classes(synthetic_df):
    with pytest.raises(dl.DataValidationError, match="both classes"):
        dl.validate_target(pd.Series(["no"] * 10))


def test_target_encoding(synthetic_df):
    X, y = dl.split_features_target(synthetic_df)
    assert set(y.unique()) == {0, 1}
    assert "y" not in X.columns
    assert (y.sum()) == (synthetic_df["y"] == "yes").sum()


def test_splits_are_disjoint_stratified_and_deterministic(synthetic_df):
    X, y = dl.split_features_target(synthetic_df)
    s1 = dl.make_splits(X, y)
    s2 = dl.make_splits(X, y)
    idx = [set(s.index) for s in (s1.X_train, s1.X_val, s1.X_test)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(X)
    assert list(s1.X_test.index) == list(s2.X_test.index)
    for part in (s1.y_train, s1.y_val, s1.y_test):
        assert abs(part.mean() - y.mean()) < 0.05


# ---- Real-data checks (skipped if the CSV is absent) ----
def test_real_dataset_shape_and_target(real_data_path):
    df = dl.load_raw(real_data_path)
    assert df.shape == (45211, 17)
    assert df["y"].value_counts().to_dict() == {"no": 39922, "yes": 5289}


def test_real_splits_preserve_prevalence(real_data_path):
    X, y = dl.split_features_target(dl.load_raw(real_data_path))
    s = dl.make_splits(X, y)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == 45211
    for part in (s.y_train, s.y_val, s.y_test):
        assert abs(part.mean() - y.mean()) < 0.002
