import os
import json
import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\ashis\Downloads\T_ONTIME_REPORTING_20260820_081428\T_ONTIME_REPORTING.csv"
OUTPUT_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"

def audit_dataset():
    print(f"Reading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    print("\n--- Basic Dataset Shape ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Remove any trailing unnamed empty column often present in BTS exports
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        print(f"Found unnamed trailing columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    print("\n--- Column Names and Types ---")
    col_info = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:3].tolist()
        col_info.append({
            "column": col,
            "dtype": dtype,
            "unique": n_unique,
            "missing": int(n_missing),
            "pct_missing": round(pct_missing, 2),
            "samples": sample_vals
        })
        print(f"{col:<25} | {dtype:<10} | Unique: {n_unique:<7} | Missing: {n_missing:<7} ({pct_missing:5.2f}%) | Sample: {sample_vals}")

    print("\n--- Date Range & Temporal Distribution ---")
    for date_col in ["FL_DATE", "FlightDate", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK"]:
        if date_col in df.columns:
            print(f"{date_col}: Min = {df[date_col].min()}, Max = {df[date_col].max()}, Unique = {df[date_col].nunique()}")

    print("\n--- Carriers & Flight Volume ---")
    carrier_cols = [c for c in ["OP_UNIQUE_CARRIER", "OP_CARRIER", "MKT_UNIQUE_CARRIER", "CARRIER", "AIRLINE_ID"] if c in df.columns]
    for cc in carrier_cols:
        print(f"Carrier col: {cc}, Count: {df[cc].nunique()}")
        print(df[cc].value_counts().head(10))

    print("\n--- Airports & Routes ---")
    if "ORIGIN" in df.columns:
        print(f"Unique Origins: {df['ORIGIN'].nunique()}")
    if "DEST" in df.columns:
        print(f"Unique Destinations: {df['DEST'].nunique()}")
    if "ORIGIN" in df.columns and "DEST" in df.columns:
        routes = df['ORIGIN'] + "->" + df['DEST']
        print(f"Unique Routes: {routes.nunique()}")

    print("\n--- Aircraft / Tail Numbers ---")
    tail_cols = [c for c in ["TAIL_NUM", "OP_CARRIER_FL_NUM"] if c in df.columns]
    for tc in tail_cols:
        print(f"{tc}: Unique = {df[tc].nunique()}, Missing = {df[tc].isnull().sum()} ({df[tc].isnull().sum()/len(df)*100:.2f}%)")

    print("\n--- Cancellations & Diversions ---")
    if "CANCELLED" in df.columns:
        print("CANCELLED counts:")
        print(df["CANCELLED"].value_counts(dropna=False))
    if "DIVERTED" in df.columns:
        print("DIVERTED counts:")
        print(df["DIVERTED"].value_counts(dropna=False))

    print("\n--- Target Candidates & Delay Statistics ---")
    delay_cols = [c for c in df.columns if "DEL" in c or "DELAY" in c]
    print(f"Delay-related columns: {delay_cols}")
    for dc in delay_cols:
        if df[dc].dtype in [np.float64, np.int64, float, int]:
            print(f"{dc:<25} | Missing: {df[dc].isnull().sum():<6} | Mean: {df[dc].mean():<8.2f} | Std: {df[dc].std():<8.2f} | Min: {df[dc].min():<6.1f} | Max: {df[dc].max():<6.1f}")
        else:
            print(f"{dc:<25} | Missing: {df[dc].isnull().sum():<6} | Distribution: {df[dc].value_counts().to_dict()}")

    if "DEP_DEL15" in df.columns:
        print("\nDEP_DEL15 Distribution (all rows):")
        print(df["DEP_DEL15"].value_counts(dropna=False, normalize=True))
        
        # When CANCELLED == 0 and DIVERTED == 0
        valid_fl = df
        if "CANCELLED" in df.columns:
            valid_fl = valid_fl[valid_fl["CANCELLED"] == 0]
        if "DIVERTED" in df.columns:
            valid_fl = valid_fl[valid_fl["DIVERTED"] == 0]
        print("\nDEP_DEL15 Distribution (completed non-diverted flights):")
        print(valid_fl["DEP_DEL15"].value_counts(dropna=False, normalize=True))
        print("Raw counts:")
        print(valid_fl["DEP_DEL15"].value_counts(dropna=False))

    # Save summary json
    audit_summary = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "columns": col_info
    }
    with open(os.path.join(OUTPUT_DIR, "audit_summary.json"), "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAudit complete. Summary written to {os.path.join(OUTPUT_DIR, 'audit_summary.json')}")

if __name__ == "__main__":
    audit_dataset()
