"""Keep the Colab notebook consistent with the project (schema, scenarios, seed) and well-formed."""
import ast
import json

import pytest

import config

NB = config.ROOT / "notebooks" / "BankSmart_AI_Colab.ipynb"
pytestmark = pytest.mark.skipif(not NB.exists(), reason="notebook not present")


def _cells():
    nb = json.loads(NB.read_text())
    return nb, [("".join(c["source"]) if isinstance(c["source"], list) else c["source"], c) for c in nb["cells"]]


def test_notebook_is_valid_and_has_no_stored_outputs():
    nb, cells = _cells()
    assert nb["nbformat"] == 4
    code = [(s, c) for s, c in cells if c["cell_type"] == "code"]
    assert code and all(not c.get("outputs") for _, c in code)
    for src, _ in code:
        ast.parse("\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!"))))


def test_notebook_schema_matches_project_config():
    _, cells = _cells()
    settings = next(s for s, c in cells if c["cell_type"] == "code" and "SCENARIO =" in s and "DOMAINS" in s)
    ns = {}
    exec(settings, ns)
    assert ns["SEED"] == config.RANDOM_SEED
    assert ns["NUMERIC"] == config.NUMERIC_FEATURES and ns["CATEGORICAL"] == config.CATEGORICAL_FEATURES
    assert ns["DOMAINS"] == config.CATEGORY_DOMAINS
    assert ns["THRESHOLD_BETA"] == config.THRESHOLD_BETA
    assert (ns["TEST_SIZE"], ns["VAL_SIZE"]) == (config.TEST_SIZE, config.VAL_SIZE)
    for sc in config.SCENARIOS:
        assert ns["scenario_features"](sc) == config.scenario_features(sc)
    assert "duration" not in ns["scenario_features"]("pre_call")


def test_notebook_states_offline_and_time_ordering_caveats():
    _, cells = _cells()
    text = " ".join(s for s, _ in cells).lower()
    assert "offline" in text and "ordered by date" in text and "roi" in text
