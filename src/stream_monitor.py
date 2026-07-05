"""Real-time fraud monitoring per the project brief: replay transactions from
the ETL database (output/fraud_detection.db) as a simulated stream, score each
one live with the trained model, and raise alerts for high-risk transactions.

This is the same file/DB-replay simulation the brief itself uses in its Spark
example (maxFilesPerTrigger=1) — no Kafka broker required. Alerts go to the
console and output/fraud_alerts.csv; email delivery via smtplib is supported
but only if SMTP_* environment variables are set (never hardcode credentials).

Mobile alerts (brief: future enhancement #3): set NTFY_TOPIC to push each alert
to your phone via ntfy.sh — install the free ntfy app, subscribe to a topic
name of your choosing, then:  NTFY_TOPIC=my-fraud-alerts python src/stream_monitor.py

Adaptive learning loop: the alert log is the analyst-review queue. Once alerts
are confirmed/rejected, appending those labeled rows to the dataset and
re-running `python src/train.py` retrains the model on the feedback, improving
accuracy over time.

Usage: python src/stream_monitor.py [--limit 500] [--threshold 0.9] [--delay 0]
"""

import argparse
import os
import smtplib
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage

import joblib
import pandas as pd

from etl import DB_PATH, TABLE
from preprocess import CATEGORICAL_COLS, FEATURE_COLS, ROOT

MODELS_DIR = ROOT / "models"
ALERTS_CSV = ROOT / "output" / "fraud_alerts.csv"


def send_alert(details, probability):
    """Alert on a high-risk transaction: console + CSV log, email if configured."""
    line = (
        f"ALERT! High-risk transaction row_id={details['row_id']} "
        f"type={details['type']} amount={details['amount']:.2f} "
        f"branch={details['branch']} p(fraud)={probability:.4f}"
    )
    print(line)

    record = {
        "alerted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_id": details["row_id"],
        "type": details["type"],
        "branch": details["branch"],
        "amount": details["amount"],
        "fraud_probability": round(probability, 6),
        "actual_fraud": details["isFraud"],
    }
    header = not ALERTS_CSV.exists()
    pd.DataFrame([record]).to_csv(ALERTS_CSV, mode="a", header=header, index=False)

    # Optional mobile push via ntfy.sh, enabled by NTFY_TOPIC. Failures must
    # not kill the stream.
    topic = os.getenv("NTFY_TOPIC")
    if topic:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=line.encode(),
                headers={"Title": "Fraud alert", "Priority": "high",
                         "Tags": "rotating_light"},
            )
            urllib.request.urlopen(req, timeout=10)
        except OSError as exc:
            print(f"  (mobile push failed: {exc})")

    # Optional email channel (brief's smtplib snippet), enabled only via env vars.
    sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("SMTP_RECIPIENT")
    if sender and password and recipient:
        msg = EmailMessage()
        msg["Subject"] = "Fraud alert: high-risk transaction"
        msg["From"], msg["To"] = sender, recipient
        msg.set_content(line)
        with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"), 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="rows to stream")
    parser.add_argument("--threshold", type=float, default=0.9, help="alert threshold")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between rows")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")

    if not DB_PATH.exists():
        raise SystemExit("Database not found — run `python src/etl.py` first.")
    conn = sqlite3.connect(DB_PATH)
    try:
        # ORDER BY step approximates arrival order of the transaction stream.
        stream = pd.read_sql(
            f'SELECT * FROM {TABLE} ORDER BY step LIMIT {args.limit}', conn
        )
    finally:
        conn.close()

    # Start each run with a fresh alert log so counts match this replay.
    ALERTS_CSV.unlink(missing_ok=True)

    print(
        f"Streaming {len(stream)} transactions from {DB_PATH.name} "
        f"(alert threshold: {args.threshold}) ...\n"
    )
    alerts = 0
    for _, row in stream.iterrows():
        X = pd.DataFrame([row[FEATURE_COLS]])
        for col in CATEGORICAL_COLS:
            X[col] = encoders[col].transform(X[col])
        probability = float(model.predict_proba(scaler.transform(X))[:, 1][0])

        if probability >= args.threshold:
            send_alert(row, probability)
            alerts += 1
        if args.delay:
            time.sleep(args.delay)

    print(f"\nStream complete: {len(stream)} transactions scored, {alerts} alerts raised.")
    if alerts:
        print(f"Alert log -> {ALERTS_CSV.relative_to(ROOT)} (analyst review queue).")
    print(
        "Adaptive learning: after analysts confirm/reject alerts, retrain with "
        "`python src/train.py` to fold the feedback into the model."
    )


if __name__ == "__main__":
    main()
