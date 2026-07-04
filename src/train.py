"""Train 5 classifiers on the preprocessed data, compare on the untouched test
set, write metrics CSV, and persist the best model by ROC-AUC.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import RANDOM_STATE, ROOT, get_data

MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# SVC with probability=True is expensive on the SMOTE-expanded train set (~54k);
# cap its training rows to keep runtime reasonable (noted in console output).
SVC_MAX_TRAIN = 20000

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=RANDOM_STATE),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
}


def main():
    X_train, X_test, y_train, y_test, _ = get_data()

    results = []
    fitted = {}
    for name, model in MODELS.items():
        X_fit, y_fit = X_train, y_train
        if name == "SVC" and len(X_train) > SVC_MAX_TRAIN:
            X_fit, _, y_fit, _ = train_test_split(
                X_train,
                y_train,
                train_size=SVC_MAX_TRAIN,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )
            print(
                f"[note] SVC trained on {SVC_MAX_TRAIN} subsampled train rows "
                f"(of {len(X_train)}) for runtime."
            )

        print(f"Training {name} ...")
        model.fit(X_fit, y_fit)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model_name": name,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            }
        )
        fitted[name] = model

    metrics_df = pd.DataFrame(results)
    print("\nModel comparison (evaluated on untouched test set):")
    print(metrics_df.to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    best_name = metrics_df.loc[metrics_df["roc_auc"].idxmax(), "model_name"]
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.pkl")
    print(f"\nBest model by roc_auc: {best_name}  ->  saved to models/best_model.pkl")

    # Held-out test confusion matrix for the best model (dashboard Page 4 reads this).
    tn, fp, fn, tp = confusion_matrix(
        y_test, fitted[best_name].predict(X_test)
    ).ravel()
    pd.DataFrame(
        [{"model_name": best_name, "tn": tn, "fp": fp, "fn": fn, "tp": tp}]
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False)
    print(f"Test confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
