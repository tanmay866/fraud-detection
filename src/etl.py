"""ETL step per the project brief: Extract the raw Excel workbook, Transform it
with the same cleaning rules as preprocess.py (labeled rows only, complete
features), and Load the cleaned table into a SQLite database.

Output: output/fraud_detection.db, table `transactions` (human-readable
categoricals + isFraud as 0/1).
"""

import sqlite3

import pandas as pd

from preprocess import DATA_PATH, FEATURE_COLS, ROOT, TARGET

DB_PATH = ROOT / "output" / "fraud_detection.db"
TABLE = "transactions"


def run_etl():
    # Extract
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

    # Transform — same row universe as training: labeled rows, complete features.
    df = df[df[TARGET].isin(["Safe", "Fraud"])].copy()
    df = df[df[FEATURE_COLS].notnull().all(axis=1)].copy()
    clean = df[FEATURE_COLS].copy()
    clean.insert(0, "row_id", df.index)
    clean["isFraud"] = (df[TARGET] == "Fraud").astype(int)

    # Load
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        clean.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.commit()
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        fraud = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE isFraud = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    print("ETL process completed.")
    print(f"Database: {DB_PATH.relative_to(ROOT)}  (table `{TABLE}`)")
    print(f"Rows loaded: {total}  |  fraud rows: {fraud}")


if __name__ == "__main__":
    run_etl()
