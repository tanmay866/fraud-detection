"""Fraud Detection Dashboard (Streamlit).

Reads the pipeline outputs and visualizes them across 5 pages. Pages 1/3/5 use
the full in-sample scored CSV for patterns/exploration; Page 4 deliberately uses
the held-out TEST metrics so performance is not overstated.

Run from repo root:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
METRICS_CSV = OUTPUT_DIR / "model_metrics.csv"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix.csv"
SCORED_CSV = OUTPUT_DIR / "scored_transactions.csv"
MODEL_FILES = {
    "best_model.pkl": MODELS_DIR / "best_model.pkl",
    "scaler.pkl": MODELS_DIR / "scaler.pkl",
    "encoders.pkl": MODELS_DIR / "encoders.pkl",
}

# Feature order the model/scaler were fit on (preprocess.py FEATURE_COLS), and the
# subset that is label-encoded. Kept local so live scoring stays self-contained.
MODEL_FEATURE_ORDER = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "unusuallogin",
    "DayOfWeek",
    "type",
    "branch",
    "Acct type",
    "Time of day",
]
MODEL_CATEGORICALS = ["type", "branch", "Acct type", "Time of day"]

# Consistent color language across the dashboard.
FRAUD_COLOR = "#d62728"  # red
LEGIT_COLOR = "#1f77b4"  # blue

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_CSV)


@st.cache_data
def load_confusion():
    return pd.read_csv(CONFUSION_CSV)


@st.cache_data
def load_scored():
    df = pd.read_csv(SCORED_CSV)
    df["day_of_week"] = df["day_of_week"].astype(int)  # 1–7, no NaN after preprocess
    return df


@st.cache_resource
def load_model_bundle():
    """Load the trained model + fitted scaler + encoders (for live scoring)."""
    import joblib

    model = joblib.load(MODEL_FILES["best_model.pkl"])
    scaler = joblib.load(MODEL_FILES["scaler.pkl"])
    encoders = joblib.load(MODEL_FILES["encoders.pkl"])
    return model, scaler, encoders


def best_row(metrics):
    """Best model row by ROC-AUC (matches train.py's selection)."""
    return metrics.loc[metrics["roc_auc"].idxmax()]


def fraud_rate_by(scored, col):
    g = scored.groupby(col)["actual_fraud"].mean().reset_index()
    g["fraud_rate_pct"] = g["actual_fraud"] * 100
    return g


def page_overview(scored, metrics):
    st.subheader("Overview")
    total = len(scored)
    fraud_count = int(scored["actual_fraud"].sum())
    fraud_rate = fraud_count / total * 100
    best = best_row(metrics)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Count", f"{fraud_count:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric(f"Model Recall ({best['model_name']})", f"{best['recall']:.1%}")

    plot_df = scored.copy()
    plot_df["Class"] = plot_df["actual_fraud"].map({0: "Legit", 1: "Fraud"})
    fig = px.histogram(
        plot_df,
        x="amount",
        color="Class",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        log_y=True,  # fraud is ~0.67% of rows; log scale keeps it visible
        color_discrete_map={"Fraud": FRAUD_COLOR, "Legit": LEGIT_COLOR},
        title="Transaction Amount Distribution — Fraud vs Legit (log count)",
    )
    st.plotly_chart(fig, width="stretch")


def page_model_comparison(metrics):
    st.subheader("Model Comparison")
    melted = metrics.melt(
        id_vars="model_name",
        value_vars=METRIC_COLS,
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        melted,
        x="model_name",
        y="score",
        color="metric",
        barmode="group",
        title="Metrics across all models (held-out test set)",
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(metrics, width="stretch")

    best = best_row(metrics)
    st.success(
        f"Best model by ROC-AUC: **{best['model_name']}** — "
        f"ROC-AUC {best['roc_auc']:.4f}, recall {best['recall']:.2f}, "
        f"precision {best['precision']:.2f}."
    )


def page_fraud_patterns(scored):
    st.subheader("Fraud Patterns")
    st.caption(
        "Fraud RATE by category across all scored transactions (full in-sample dataset)."
    )

    for col, label in [
        ("type", "Transaction Type"),
        ("acct_type", "Account Type"),
        ("time_of_day", "Time of Day"),
    ]:
        g = fraud_rate_by(scored, col).sort_values("fraud_rate_pct", ascending=False)
        fig = px.bar(
            g,
            x=col,
            y="fraud_rate_pct",
            title=f"Fraud rate by {label}",
            color_discrete_sequence=[FRAUD_COLOR],
        )
        fig.update_yaxes(title="Fraud rate (%)")
        st.plotly_chart(fig, width="stretch")

    g = fraud_rate_by(scored, "day_of_week")
    fig = px.bar(
        g,
        x="day_of_week",
        y="fraud_rate_pct",
        title="Fraud rate by Day of Week",
        color_discrete_sequence=[FRAUD_COLOR],
    )
    fig.update_yaxes(title="Fraud rate (%)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    pivot = (
        scored.pivot_table(
            index="day_of_week",
            columns="time_of_day",
            values="actual_fraud",
            aggfunc="mean",
        )
        * 100
    )
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="Reds",
        aspect="auto",
        title="Fraud rate (%) — Day of Week vs Time of Day",
        labels={"color": "Fraud rate (%)"},
    )
    st.plotly_chart(fig, width="stretch")


def page_confusion(confusion, metrics):
    st.subheader("Confusion Matrix — Held-out test performance")
    st.caption(
        "These are TEST-SET results — rows the model never trained on — not the "
        "full in-sample dataset shown on the Overview, Fraud Patterns, and Explorer pages."
    )

    row = confusion.iloc[0]
    tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])

    z = [[tn, fp], [fn, tp]]
    labels = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted: Legit", "Predicted: Fraud"],
            y=["Actual: Legit", "Actual: Fraud"],
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{row['model_name']} — test-set confusion matrix")
    st.plotly_chart(fig, width="stretch")

    best = best_row(metrics)
    c1, c2 = st.columns(2)
    c1.metric("Precision (test)", f"{best['precision']:.1%}")
    c2.metric("Recall (test)", f"{best['recall']:.1%}")

    st.markdown(
        f"- **False Positives (FP = {fp}):** legit transactions wrongly flagged — "
        "each one is a blocked, frustrated genuine customer."
    )
    st.markdown(
        f"- **False Negatives (FN = {fn}):** fraud the model missed — "
        "each one is undetected financial loss to the bank."
    )


def page_explorer(scored):
    st.subheader("Transaction Explorer")

    threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01)
    flagged = scored["fraud_probability"] >= threshold
    st.metric(
        "Flagged at this threshold",
        f"{int(flagged.sum()):,}",
        help="flagged = fraud_probability >= threshold",
    )
    st.caption(
        f"At threshold {threshold:.2f}, "
        f"{int(flagged.sum()):,} of {len(scored):,} transactions are flagged as fraud."
    )

    c1, c2, c3 = st.columns(3)
    types = c1.multiselect("Type", sorted(scored["type"].unique()))
    accts = c2.multiselect("Account type", sorted(scored["acct_type"].unique()))
    tods = c3.multiselect("Time of day", sorted(scored["time_of_day"].unique()))

    amount_min, amount_max = float(scored["amount"].min()), float(scored["amount"].max())
    lo, hi = st.slider("Amount range", amount_min, amount_max, (amount_min, amount_max))

    view = scored.copy()
    view["flagged"] = flagged.astype(int)
    if types:
        view = view[view["type"].isin(types)]
    if accts:
        view = view[view["acct_type"].isin(accts)]
    if tods:
        view = view[view["time_of_day"].isin(tods)]
    view = view[(view["amount"] >= lo) & (view["amount"] <= hi)]
    view = view.sort_values("fraud_probability", ascending=False)

    st.caption(f"{len(view):,} transactions match the filters.")
    st.dataframe(view, width="stretch")

    render_manual_scoring(scored)


def render_manual_scoring(scored):
    """Isolated: score a single manually-entered transaction with the live model.

    The form collects 5 fields; the other 7 model features are filled with
    dataset-typical defaults so the full 12-feature vector matches training.
    """
    st.divider()
    st.markdown("### Score a New Transaction")
    st.caption(
        "Enter a transaction to score it live with the trained model. "
        "Fields not shown use dataset-typical defaults."
    )

    missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing:
        st.info(
            "Model files not found (" + ", ".join(missing) + "). "
            "Run `python src/train.py` from the repo root to enable live scoring."
        )
        return

    model, scaler, encoders = load_model_bundle()

    with st.form("score_new_transaction"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        txn_type = c2.selectbox("Type", list(encoders["type"].classes_))
        acct_type = c3.selectbox("Account type", list(encoders["Acct type"].classes_))
        c4, c5 = st.columns(2)
        time_of_day = c4.selectbox("Time of day", list(encoders["Time of day"].classes_))
        day_of_week = c5.selectbox("Day of week", [1, 2, 3, 4, 5, 6, 7])
        submitted = st.form_submit_button("Score transaction")

    if not submitted:
        return

    # Defaults for the 7 features not on the form, from the scored dataset.
    default_branch = scored["branch"].mode().iloc[0]
    row = {
        "step": scored["step"].median(),
        "amount": amount,
        "oldbalanceOrg": scored["oldbalanceOrg"].median(),
        "newbalanceOrig": scored["newbalanceOrig"].median(),
        "oldbalanceDest": scored["oldbalanceDest"].median(),
        "newbalanceDest": scored["newbalanceDest"].median(),
        "unusuallogin": scored["unusuallogin"].median(),
        "DayOfWeek": float(day_of_week),
        "type": txn_type,
        "branch": default_branch,
        "Acct type": acct_type,
        "Time of day": time_of_day,
    }

    # Transform exactly as preprocess.py does: encode categoricals, then scale.
    X = pd.DataFrame([row], columns=MODEL_FEATURE_ORDER)
    for col in MODEL_CATEGORICALS:
        X[col] = encoders[col].transform(X[col])
    proba = float(model.predict_proba(scaler.transform(X))[:, 1][0])
    flagged = proba >= 0.5

    m1, m2 = st.columns(2)
    m1.metric("Fraud probability", f"{proba:.1%}")
    m2.metric("Decision @ 0.5", "🚩 FLAGGED" if flagged else "✅ Not flagged")
    if flagged:
        st.error(f"Flagged as fraud — probability {proba:.1%} ≥ 0.50.")
    else:
        st.success(f"Not flagged — probability {proba:.1%} < 0.50.")
    st.caption(
        f"Defaults used for non-entered features: branch = {default_branch}, "
        f"step = {row['step']:.0f}, balances & unusuallogin at dataset medians."
    )


def main():
    st.title("💳 Fraud Detection Dashboard")

    missing = [
        p.name for p in (METRICS_CSV, CONFUSION_CSV, SCORED_CSV) if not p.exists()
    ]
    if missing:
        st.error(
            "Missing required output files: "
            + ", ".join(missing)
            + ".\n\nRun the pipeline first from the repo root:\n\n"
            "```\npython src/train.py\npython src/score.py\n```"
        )
        st.stop()

    metrics = load_metrics()
    confusion = load_confusion()
    scored = load_scored()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Model Comparison",
            "Fraud Patterns",
            "Confusion Matrix",
            "Transaction Explorer",
        ],
    )

    if page == "Overview":
        page_overview(scored, metrics)
    elif page == "Model Comparison":
        page_model_comparison(metrics)
    elif page == "Fraud Patterns":
        page_fraud_patterns(scored)
    elif page == "Confusion Matrix":
        page_confusion(confusion, metrics)
    else:
        page_explorer(scored)


if __name__ == "__main__":
    main()
