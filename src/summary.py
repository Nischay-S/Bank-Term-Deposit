"""Build small, app-ready summaries so the dashboard runs without the raw dataset.

    python -m src.summary

Writes (all derived from the public UCI Bank Marketing data, CC BY 4.0):
  artifacts/dataset_summary.json   - counts, prevalence, subscription rate by category
  artifacts/background_sample.csv  - 500 TRAINING rows used as a SHAP/occlusion reference
  artifacts/demo_customers.csv     - 200 VALIDATION rows (with historical 'y') for demos
"""
import json
import logging

import pandas as pd

import config
from src import data_loader as dl

logger = logging.getLogger(__name__)
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def build_summary():
    df = dl.load_raw()
    X, y = dl.split_features_target(df)
    s = dl.make_splits(X, y)
    rates = {}
    for col in config.CATEGORICAL_FEATURES:
        g = df.assign(_y=y).groupby(col)["_y"].agg(["size", "mean"]).reset_index()
        g.columns = ["category", "n", "rate"]
        if col == "month":
            g["category"] = pd.Categorical(g["category"], MONTH_ORDER, ordered=True)
            g = g.sort_values("category")
            g["category"] = g["category"].astype(str)
        else:
            g = g.sort_values("n", ascending=False)
        rates[col] = g.to_dict(orient="records")
    summary = {
        "n_rows": int(len(df)), "n_input_features": len(config.ALL_FEATURES),
        "n_positive": int(y.sum()), "prevalence": float(y.mean()),
        "split_sizes": {"train": len(s.X_train), "validation": len(s.X_val), "test": len(s.X_test)},
        "unknown_share": {c: float((df[c] == "unknown").mean()) for c in ("job", "education", "contact", "poutcome")},
        "never_previously_contacted_share": float((df["pdays"] == -1).mean()),
        "rates_by_category": rates,
        "numeric_summary": df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
        "source": "UCI Machine Learning Repository, Bank Marketing (dataset 222), CC BY 4.0; Moro, Cortez & Rita (2014)",
    }
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    (config.ARTIFACT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    s.X_train.sample(500, random_state=config.RANDOM_SEED).to_csv(config.ARTIFACT_DIR / "background_sample.csv", index=False)
    demo = s.X_val.sample(200, random_state=config.RANDOM_SEED).assign(y=s.y_val.map({1: "yes", 0: "no"}))
    demo.to_csv(config.ARTIFACT_DIR / "demo_customers.csv", index=False)
    logger.info("Summary artifacts written (%d rows, prevalence %.4f)", len(df), y.mean())
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_summary()
