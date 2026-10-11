"""Dataset loading, schema/target validation and stratified splitting."""
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema or target."""


def _detect_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    return ";" if header.count(";") > header.count(",") else ","


def validate_schema(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        raise DataValidationError("Dataset is empty.")
    expected = config.ALL_FEATURES + [config.TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )
    for col in config.NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric.")
    for col, domain in config.CATEGORY_DOMAINS.items():
        bad = set(df[col].dropna().unique()) - domain
        if bad:
            raise DataValidationError(f"Column '{col}' has unexpected values: {sorted(bad)}")
    n_null = int(df[expected].isna().sum().sum())
    if n_null:
        logger.warning("Dataset contains %d missing cells; the pipeline imputes them.", n_null)


def validate_target(y: pd.Series) -> None:
    values = set(y.dropna().astype(str).str.strip().str.lower().unique())
    allowed = {config.POSITIVE_LABEL, config.NEGATIVE_LABEL}
    if y.isna().any():
        raise DataValidationError("Target contains missing values.")
    if not values <= allowed:
        raise DataValidationError(f"Target has invalid values {sorted(values - allowed)}; expected {sorted(allowed)}.")
    if values != allowed:
        raise DataValidationError("Target must contain both classes.")


def load_raw(path=None) -> pd.DataFrame:
    """Load and validate the raw CSV (comma or semicolon separated)."""
    path = Path(path) if path else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    df = pd.read_csv(path, sep=_detect_separator(path))
    validate_schema(df)
    validate_target(df[config.TARGET])
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) with y encoded as 1 = subscribed, 0 = not subscribed."""
    validate_target(df[config.TARGET])
    y = (df[config.TARGET].astype(str).str.strip().str.lower() == config.POSITIVE_LABEL).astype(int)
    X = df[config.ALL_FEATURES].copy()
    return X, y


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def make_splits(X, y, seed=config.RANDOM_SEED, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE) -> Splits:
    """Stratified random train/validation/test split (rows assumed i.i.d.)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def read_uploaded_csv(file_like, filename: str = "upload.csv",
                      max_mb: float = config.MAX_UPLOAD_MB, max_rows: int = config.MAX_UPLOAD_ROWS) -> pd.DataFrame:
    """Safely parse a user-uploaded CSV (comma or semicolon). Raises DataValidationError with a clear message."""
    import io
    if not str(filename).lower().endswith(".csv"):
        raise DataValidationError(f"'{filename}' is not a .csv file. Upload a CSV exported from Excel or a database.")
    raw = file_like.read() if hasattr(file_like, "read") else bytes(file_like)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) == 0:
        raise DataValidationError("The uploaded file is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DataValidationError(f"File is {len(raw) / 1024 / 1024:.1f} MB; the limit is {max_mb:g} MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DataValidationError("The file is not UTF-8 text. Re-save it as 'CSV UTF-8'.") from None
    header = text.splitlines()[0] if text.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    # pandas silently renames duplicate headers ('age', 'age.1'), so check the raw header line first
    import csv
    names = next(csv.reader([header], delimiter=sep), [])
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise DataValidationError(f"Duplicate column names: {dupes}. Rename or remove them and upload again.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:  # pandas raises several parser error types
        raise DataValidationError(f"Could not read the CSV: {exc}") from None
    if df.empty:
        raise DataValidationError("The file has a header but no data rows.")
    if len(df) > max_rows:
        raise DataValidationError(f"File has {len(df):,} rows; the limit is {max_rows:,}. Split it into smaller files.")
    def _numeric(name):
        try:
            float(str(name))
            return True
        except ValueError:
            return False
    if all(str(c).startswith("Unnamed") or _numeric(c) for c in df.columns):
        raise DataValidationError("No header row found. The first line must contain the column names, not data.")
    return df
