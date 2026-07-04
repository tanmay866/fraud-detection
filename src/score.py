"""Score every usable transaction with the trained best model and export a
human-readable scored CSV for the dashboard.

Features are rebuilt exactly as preprocess does, but encoders and scaler are
loaded and applied with transform only — never refit (that would leak/​drift).
"""

import joblib
import pandas as pd

from preprocess import DATA_PATH, FEATURE_COLS, ROOT, TARGET

MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"


def main():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")

    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

    # Same row universe as training: labeled rows, complete features.
    df = df[df[TARGET].isin(["Safe", "Fraud"])].copy()
    df = df[df[FEATURE_COLS].notnull().all(axis=1)].copy()

    # Encode a copy for the model; keep df's raw categoricals for the export.
    X = df[FEATURE_COLS].copy()
    for col, encoder in encoders.items():
        X[col] = encoder.transform(X[col])
    X_scaled = scaler.transform(X)

    predicted = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]

    scored = pd.DataFrame(
        {
            "row_id": df.index,
            "step": df["step"].to_numpy(),
            "type": df["type"].to_numpy(),
            "branch": df["branch"].to_numpy(),
            "amount": df["amount"].to_numpy(),
            "oldbalanceOrg": df["oldbalanceOrg"].to_numpy(),
            "newbalanceOrig": df["newbalanceOrig"].to_numpy(),
            "oldbalanceDest": df["oldbalanceDest"].to_numpy(),
            "newbalanceDest": df["newbalanceDest"].to_numpy(),
            "unusuallogin": df["unusuallogin"].to_numpy(),
            "acct_type": df["Acct type"].to_numpy(),
            "time_of_day": df["Time of day"].to_numpy(),
            "day_of_week": df["DayOfWeek"].to_numpy(),
            "actual_fraud": (df[TARGET] == "Fraud").astype(int).to_numpy(),
            "predicted_fraud": predicted,
            "fraud_probability": probability.round(6),
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    scored.to_csv(OUTPUT_DIR / "scored_transactions.csv", index=False)

    print(f"Scored {len(scored)} rows.")
    print(f"Predicted fraud (flagged): {int(scored['predicted_fraud'].sum())}")
    print(f"Actual fraud:              {int(scored['actual_fraud'].sum())}")


if __name__ == "__main__":
    main()
