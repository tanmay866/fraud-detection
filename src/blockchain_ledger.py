"""Blockchain-based fraud prevention (brief: future enhancement #2, demo form).

Records every model-flagged transaction in a SHA-256 hash-chained ledger inside
the project database: each block stores the hash of the previous block, so any
retroactive edit breaks the chain and is detected by verification. This is the
tamper-evidence property of a blockchain without the distributed network.

Usage: python src/blockchain_ledger.py          # build ledger + verify + tamper demo
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from etl import DB_PATH
from preprocess import ROOT

SCORED_CSV = ROOT / "output" / "scored_transactions.csv"
BATCH_SIZE = 10


def block_hash(index, timestamp, data, prev_hash):
    payload = f"{index}|{timestamp}|{data}|{prev_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()


def build_ledger(conn, flagged):
    conn.execute("DROP TABLE IF EXISTS ledger")
    conn.execute(
        "CREATE TABLE ledger (block_index INTEGER PRIMARY KEY, "
        "timestamp TEXT, data TEXT, prev_hash TEXT, hash TEXT)"
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_block(index, data, prev_hash):
        h = block_hash(index, now, data, prev_hash)
        conn.execute("INSERT INTO ledger VALUES (?, ?, ?, ?, ?)",
                     (index, now, data, prev_hash, h))
        return h

    prev = add_block(0, "GENESIS", "0" * 64)
    for i in range(0, len(flagged), BATCH_SIZE):
        batch = flagged.iloc[i : i + BATCH_SIZE]
        data = json.dumps(batch.to_dict(orient="records"), separators=(",", ":"))
        prev = add_block(i // BATCH_SIZE + 1, data, prev)
    conn.commit()


def verify_ledger(conn):
    """Recompute every hash and check the chain links; return list of bad blocks."""
    rows = conn.execute(
        "SELECT block_index, timestamp, data, prev_hash, hash "
        "FROM ledger ORDER BY block_index"
    ).fetchall()
    bad, expected_prev = [], "0" * 64
    for index, timestamp, data, prev_hash, stored in rows:
        if prev_hash != expected_prev or block_hash(
            index, timestamp, data, prev_hash
        ) != stored:
            bad.append(index)
        expected_prev = stored
    return bad


def main():
    if not SCORED_CSV.exists():
        raise SystemExit("Run `python src/score.py` first.")
    if not DB_PATH.exists():
        raise SystemExit("Run `python src/etl.py` first.")

    scored = pd.read_csv(SCORED_CSV)
    flagged = scored[scored["predicted_fraud"] == 1][
        ["row_id", "type", "branch", "amount", "fraud_probability"]
    ]

    conn = sqlite3.connect(DB_PATH)
    try:
        build_ledger(conn, flagged)
        n_blocks = conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
        print(f"Ledger built: {len(flagged)} flagged transactions in {n_blocks} blocks "
              f"(table `ledger` in {DB_PATH.name}).")

        bad = verify_ledger(conn)
        print(f"Verification: {'chain VALID' if not bad else f'TAMPERED blocks {bad}'}")

        # Tamper demo: silently lower an amount inside block 1, then re-verify.
        print("\nTamper demo — editing a recorded amount inside block 1 ...")
        conn.execute(
            "UPDATE ledger SET data = replace(data, '\"amount\":', '\"amount\":0.0') "
            "WHERE block_index = 1"
        )
        bad = verify_ledger(conn)
        print(f"Verification: {'chain VALID' if not bad else f'TAMPERING DETECTED at block(s) {bad}'}")

        print("\nRestoring a clean ledger ...")
        build_ledger(conn, flagged)
        bad = verify_ledger(conn)
        print(f"Verification: {'chain VALID' if not bad else f'TAMPERED blocks {bad}'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
