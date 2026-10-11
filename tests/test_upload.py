import io

import pytest

import config
from src import data_loader as dl


def _csv(df, sep=","):
    return io.BytesIO(df.to_csv(index=False, sep=sep).encode("utf-8"))


def test_valid_upload_comma_and_semicolon(synthetic_df):
    for sep in (",", ";"):
        assert dl.read_uploaded_csv(_csv(synthetic_df, sep), "a.csv").shape == synthetic_df.shape


def test_rejects_wrong_extension(synthetic_df):
    with pytest.raises(dl.DataValidationError, match=".csv"):
        dl.read_uploaded_csv(_csv(synthetic_df), "data.xlsx")


def test_rejects_empty_file():
    with pytest.raises(dl.DataValidationError, match="empty"):
        dl.read_uploaded_csv(io.BytesIO(b""), "a.csv")


def test_rejects_header_only(synthetic_df):
    with pytest.raises(dl.DataValidationError, match="no data rows"):
        dl.read_uploaded_csv(_csv(synthetic_df.head(0)), "a.csv")


def test_rejects_oversize_file(synthetic_df):
    with pytest.raises(dl.DataValidationError, match="limit"):
        dl.read_uploaded_csv(_csv(synthetic_df), "a.csv", max_mb=0.0001)


def test_rejects_too_many_rows(synthetic_df):
    with pytest.raises(dl.DataValidationError, match="rows"):
        dl.read_uploaded_csv(_csv(synthetic_df), "a.csv", max_rows=10)


def test_rejects_non_utf8():
    with pytest.raises(dl.DataValidationError, match="UTF-8"):
        dl.read_uploaded_csv(io.BytesIO("a,b\n\xe9,1".encode("latin-1")), "a.csv")


def test_rejects_duplicate_columns():
    with pytest.raises(dl.DataValidationError, match="Duplicate"):
        dl.read_uploaded_csv(io.BytesIO(b"age,age\n1,2\n"), "a.csv")


def test_rejects_missing_header():
    with pytest.raises(dl.DataValidationError):
        dl.read_uploaded_csv(io.BytesIO(b"1,2\n3,4\n"), "a.csv")


def test_limits_come_from_config():
    assert config.MAX_UPLOAD_MB > 0 and config.MAX_UPLOAD_ROWS > 0
