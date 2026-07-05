"""AI-driven customer risk scoring (brief: future enhancement #1, demo form).

Aggregates the trained model's fraud probabilities per customer (nameOrig) with
behavioral signals into a 0-100 risk score and an A-E creditworthiness-style
tier. True credit scoring needs credit-history data this dataset doesn't have;
this is the fraud-risk half of that picture, driven by the live model.

Reads output/scored_transactions.csv (join on row_id = original Excel row) and
exports output/customer_risk_scores.csv.
"""

import pandas as pd

from preprocess import DATA_PATH, ROOT, TARGET

SCORED_CSV = ROOT / "output" / "scored_transactions.csv"
OUTPUT_CSV = ROOT / "output" / "customer_risk_scores.csv"

TIERS = [(20, "A"), (40, "B"), (60, "C"), (80, "D"), (101, "E")]


def tier(score):
    return next(t for cutoff, t in TIERS if score < cutoff)


def main():
    if not SCORED_CSV.exists():
        raise SystemExit("Run `python src/score.py` first.")
    scored = pd.read_csv(SCORED_CSV)

    names = pd.read_excel(DATA_PATH, sheet_name="Sheet1")[["nameOrig", TARGET]]
    scored["customer_id"] = scored["row_id"].map(names["nameOrig"])
    scored = scored[scored["customer_id"].notnull()]

    # Behavioral signals, each normalized to 0-1.
    scored["amount_pctl"] = scored["amount"].rank(pct=True)
    scored["login_risk"] = (scored["unusuallogin"] / 10).clip(upper=1.0)

    per_customer = scored.groupby("customer_id").agg(
        n_transactions=("row_id", "size"),
        total_amount=("amount", "sum"),
        max_fraud_probability=("fraud_probability", "max"),
        amount_pctl=("amount_pctl", "max"),
        login_risk=("login_risk", "max"),
        confirmed_fraud=("actual_fraud", "max"),
    )

    # Model probability dominates; behavior and confirmed history refine it.
    per_customer["risk_score"] = (
        100
        * (
            0.6 * per_customer["max_fraud_probability"]
            + 0.2 * per_customer["amount_pctl"]
            + 0.1 * per_customer["login_risk"]
            + 0.1 * per_customer["confirmed_fraud"]
        )
    ).round(2)
    per_customer["risk_tier"] = per_customer["risk_score"].map(tier)

    out = per_customer.sort_values("risk_score", ascending=False).reset_index()
    out.drop(columns=["amount_pctl", "login_risk"]).to_csv(OUTPUT_CSV, index=False)

    print(f"Scored {len(out):,} customers -> {OUTPUT_CSV.relative_to(ROOT)}")
    print("\nTier distribution (A = most creditworthy, E = highest risk):")
    print(out["risk_tier"].value_counts().sort_index().to_string())
    print("\nTop 10 highest-risk customers:")
    cols = ["customer_id", "n_transactions", "total_amount",
            "max_fraud_probability", "risk_score", "risk_tier"]
    print(out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
