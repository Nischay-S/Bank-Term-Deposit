# Data

Expected file: `data/raw/bank_marketing.csv` (UCI Bank Marketing, 17-column "bank-full" version;
45,211 rows). Source: UCI Machine Learning Repository, dataset 222 (CC BY 4.0).
Original study: Moro, Cortez & Rita (2014).

The CSV is git-ignored. Do NOT use the 4,521-row sample `test.csv`/`bank.csv` for evaluation:
every one of its rows also appears in the full file (verified), so it would leak training data.
