"""Graph-based fraud network analysis per the project brief: treat accounts as
nodes and transactions (nameOrig -> nameDest) as edges, find connected
components with union-find, and surface the networks and destination accounts
that fraud concentrates in.

Origin accounts are ~all unique, so networks form around shared destination
accounts. Exports output/graph_fraud_networks.csv (one row per fraud-touching
network) and prints the highest-risk destination accounts.
"""

import pandas as pd

from preprocess import DATA_PATH, FEATURE_COLS, ROOT, TARGET

OUTPUT_CSV = ROOT / "output" / "graph_fraud_networks.csv"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

    # Same row universe as training: labeled rows, complete features.
    df = df[df[TARGET].isin(["Safe", "Fraud"])].copy()
    df = df[df[FEATURE_COLS].notnull().all(axis=1)].copy()

    # nameOrig/nameDest are not model features, so the filter above doesn't
    # cover them — a NaN node id would loop union-find forever (NaN != NaN).
    n_before = len(df)
    df = df[df["nameOrig"].notnull() & df["nameDest"].notnull()].copy()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing account ids.")

    df["is_fraud"] = (df[TARGET] == "Fraud").astype(int)

    uf = UnionFind()
    for orig, dest in zip(df["nameOrig"], df["nameDest"]):
        uf.union(orig, dest)

    df["network_id"] = [uf.find(o) for o in df["nameOrig"]]

    # Vectorized aggregation only — per-group Python lambdas are minutes-slow
    # across ~6.5k networks.
    df["fraud_amount"] = df["amount"] * df["is_fraud"]
    networks = df.groupby("network_id").agg(
        n_transactions=("is_fraud", "size"),
        n_fraud=("is_fraud", "sum"),
        fraud_amount=("fraud_amount", "sum"),
    )
    members = pd.concat(
        [
            df[["network_id", "nameOrig"]].rename(columns={"nameOrig": "account"}),
            df[["network_id", "nameDest"]].rename(columns={"nameDest": "account"}),
        ]
    )
    networks["accounts"] = members.groupby("network_id")["account"].nunique()
    networks = networks.reset_index()

    fraud_networks = (
        networks[networks["n_fraud"] > 0]
        .sort_values(["n_fraud", "fraud_amount"], ascending=False)
        .reset_index(drop=True)
    )
    fraud_networks.insert(0, "network_rank", fraud_networks.index + 1)
    fraud_networks.to_csv(OUTPUT_CSV, index=False)

    print(f"Accounts (nodes):          {pd.concat([df['nameOrig'], df['nameDest']]).nunique():,}")
    print(f"Transactions (edges):      {len(df):,}")
    print(f"Networks (components):     {networks.shape[0]:,}")
    print(f"Networks containing fraud: {len(fraud_networks):,}")
    print(f"Exported -> {OUTPUT_CSV.relative_to(ROOT)}")

    multi = fraud_networks[fraud_networks["n_fraud"] > 1]
    if len(multi):
        print(f"\nNetworks with MULTIPLE frauds (organized-fraud signal): {len(multi)}")
        print(multi.head(10).to_string(index=False))

    dest_fraud = (
        df[df["is_fraud"] == 1]
        .groupby("nameDest")
        .agg(fraud_received=("is_fraud", "size"), fraud_amount=("amount", "sum"))
        .sort_values("fraud_received", ascending=False)
    )
    repeat = dest_fraud[dest_fraud["fraud_received"] > 1]
    print(f"\nDestination accounts receiving fraud: {len(dest_fraud)}")
    if len(repeat):
        print(f"Repeat fraud destinations (mule-account signal): {len(repeat)}")
        print(repeat.head(10).to_string())
    else:
        print("No destination account received more than one fraudulent transaction.")


if __name__ == "__main__":
    main()
