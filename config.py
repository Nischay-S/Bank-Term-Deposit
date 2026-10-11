"""Central configuration for BankSmart AI. All paths are relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank_marketing.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20   # fraction of all data held out as the final test set
VAL_SIZE = 0.20    # fraction of all data used for validation (model/threshold selection)

# Target
TARGET = "y"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"

# Schema (UCI Bank Marketing, 17-column "bank-full" version)
NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categorical columns whose value sets are fixed by the dataset documentation.
# (job is omitted: its list is long and checked only for being non-numeric.)
CATEGORY_DOMAINS = {
    "marital": {"married", "single", "divorced"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "default": {"yes", "no"},
    "housing": {"yes", "no"},
    "loan": {"yes", "no"},
    "contact": {"cellular", "telephone", "unknown"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "poutcome": {"failure", "other", "success", "unknown"},
}

# Scenarios. 'duration' is only known AFTER a call ends, so it is excluded pre-call.
LEAKAGE_FEATURES = {"pre_call": ["duration"], "post_call": []}
SCENARIOS = ("pre_call", "post_call")


def scenario_features(scenario: str) -> list:
    """Raw input columns allowed for a scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {SCENARIOS}.")
    banned = set(LEAKAGE_FEATURES[scenario])
    return [c for c in ALL_FEATURES if c not in banned]

# ---- Modelling / selection settings ----
CV_FOLDS = 5
SELECTION_METRIC = "pr_auc"   # chosen on VALIDATION data; the test split is never used for selection
# Threshold objective: maximise F-beta on validation data. beta=2 weights recall twice as much as
# precision, i.e. ASSUMES a missed likely subscriber costs more than one extra call. No cost data is
# available, so this is an explicit assumption that the app lets the user override.
THRESHOLD_BETA = 2.0
BOOTSTRAP_ROUNDS = 1000

# ---- App / upload limits ----
MAX_UPLOAD_MB = 25
MAX_UPLOAD_ROWS = 200_000
