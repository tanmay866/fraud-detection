"""Load → filter → clean → encode → split → scale → SMOTE for the fraud dataset.

Source: data/Fraud_Detection.xlsx. Target `isFraud` is Safe/Fraud/Not reviewed;
we keep only Safe/Fraud (Fraud=1, Safe=0). Fraud is ~0.67% of rows, so the
split → fit scaler on train → SMOTE on train order is critical: SMOTE or scaler
fit touching the test set leaks and fakes metrics.
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Fraud_Detection.xlsx"
MODELS_DIR = ROOT / "models"

TARGET = "isFraud"
NUMERIC_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "unusuallogin",
    "DayOfWeek",
]
CATEGORICAL_COLS = ["type", "branch", "Acct type", "Time of day"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
RANDOM_STATE = 42


def get_data(verbose=False):
    """Return X_train, X_test, y_train, y_test, feature_names.

    Fits and persists encoders.pkl and scaler.pkl to models/. SMOTE is applied
    to the training split only.
    """
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

    # Keep only labeled rows; drop the "Not reviewed" cases, then map to 0/1.
    df = df[df[TARGET].isin(["Safe", "Fraud"])].copy()
    y = (df[TARGET] == "Fraud").astype(int)

    # Keep only the 12 features (discards Column1, isFraud - Copy, nameOrig,
    # nameDest, isFlaggedFraud, Date of transaction, DayOfWeek(new)).
    X = df[FEATURE_COLS].copy()

    # Drop rows with any missing feature value (verified: no fraud rows affected).
    complete = X.notnull().all(axis=1)
    X, y = X[complete], y[complete]

    # Label-encode categoricals; persist the fitted encoders for score.py / app.py.
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Fit scaler on TRAIN only, then transform both splits.
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    if verbose:
        print(f"Rows after filtering + dropna: {len(X) + 0}")
        print(f"Train shape (pre-SMOTE):  {X_train.shape}")
        print(f"Test shape:               {X_test.shape}")
        print("Train class balance (pre-SMOTE):")
        print(y_train.value_counts().sort_index().to_string())
        print("Test class balance:")
        print(y_test.value_counts().sort_index().to_string())

    # SMOTE on TRAIN only — never on the test set.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    if verbose:
        print(f"\nTrain shape (post-SMOTE): {X_train.shape}")
        print("Train class balance (post-SMOTE):")
        print(pd.Series(y_train).value_counts().sort_index().to_string())

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    get_data(verbose=True)
