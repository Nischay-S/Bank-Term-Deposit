"""Offline campaign-prioritisation analysis (historical test data only).

OFFLINE ONLY: these numbers describe how well the model ranked customers in a historical hold-out set.
They do not show that contacting the top-ranked customers would CAUSE extra subscriptions, and no
revenue, cost or ROI figure can be derived without cost/revenue data and a randomised experiment.
"""
import argparse
import logging
import math

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

OFFLINE_DISCLAIMER = (
    "Offline evaluation on historical data. A high lift means the model ranked past subscribers near the top; "
    "it does not prove that contacting these customers will create additional subscriptions, and it says nothing "
    "about revenue or ROI. Measuring real uplift needs a controlled experiment.")


def rank_customers(y_true, proba) -> pd.DataFrame:
    df = pd.DataFrame({"probability": np.asarray(proba, dtype=float), "actual": np.asarray(y_true, dtype=int)},
                      index=getattr(y_true, "index", None))
    df = df.sort_values("probability", ascending=False, kind="mergesort")
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _check_fraction(f):
    if not (0 < f <= 1):
        raise ValueError(f"Outreach fraction must be in (0, 1], got {f}.")


def campaign_table(y_true, proba, fractions=(0.10, 0.20, 0.30)) -> pd.DataFrame:
    ranked = rank_customers(y_true, proba)
    n, total_pos = len(ranked), int(ranked.actual.sum())
    if total_pos == 0:
        raise ValueError("No positive customers in the evaluation set.")
    prevalence = total_pos / n
    rows = []
    for f in fractions:
        _check_fraction(f)
        k = int(math.ceil(f * n))
        top = ranked.iloc[:k]
        found = int(top.actual.sum())
        precision = found / k
        rows.append({
            "outreach_fraction": f, "customers_selected": k, "subscribers_found": found,
            "precision": precision, "recall": found / total_pos, "lift": precision / prevalence,
            "random_expected_subscribers": k * prevalence, "random_expected_recall": k / n,
            "extra_vs_random": found - k * prevalence, "min_probability_selected": float(top.probability.iloc[-1]),
            "test_prevalence": prevalence,
        })
    return pd.DataFrame(rows)


def gains_curve(y_true, proba, n_points=100) -> pd.DataFrame:
    ranked = rank_customers(y_true, proba)
    n, total_pos = len(ranked), int(ranked.actual.sum())
    cum = ranked.actual.cumsum().to_numpy()
    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    prev = total_pos / n
    return pd.DataFrame({"fraction_contacted": ks / n, "recall": cum[ks - 1] / total_pos,
                         "precision": cum[ks - 1] / ks, "lift": cum[ks - 1] / ks / prev,
                         "random_recall": ks / n})


def select_top_fraction(scored: pd.DataFrame, fraction: float, proba_col="probability") -> pd.DataFrame:
    """Top share of customers by predicted probability (for FUTURE customers: no outcomes needed)."""
    _check_fraction(fraction)
    k = int(math.ceil(fraction * len(scored)))
    return scored.sort_values(proba_col, ascending=False, kind="mergesort").head(k)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(config.SCENARIOS) + ["all"], default="all")
    args = ap.parse_args()
    for sc in (config.SCENARIOS if args.scenario == "all" else [args.scenario]):
        d = config.ARTIFACT_DIR / sc
        preds = pd.read_csv(d / "test_predictions.csv", index_col=0)
        campaign_table(preds.y_true, preds.proba).to_csv(d / "campaign_table_test.csv", index=False)
        gains_curve(preds.y_true, preds.proba).to_csv(d / "gains_curve_test.csv", index=False)
        logger.info("[%s] campaign tables written", sc)


if __name__ == "__main__":
    main()
