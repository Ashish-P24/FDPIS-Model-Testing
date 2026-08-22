import os
import sys
import gc
import json
import joblib
import numpy as np
import pandas as pd

DATA_PATH = r"C:\Users\ashis\Downloads\T_ONTIME_REPORTING_20260820_081428\T_ONTIME_REPORTING.csv"
OUTPUT_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"
MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_clean_raw_data(data_path=DATA_PATH):
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Drop unnamed columns
    unnamed = [c for c in df.columns if "Unnamed" in c]
    if unnamed:
        df = df.drop(columns=unnamed)
        
    print(f"Initial raw shape: {df.shape}")
    
    # Filter out cancelled and diverted flights
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0].copy()
    if "DIVERTED" in df.columns:
        df = df[df["DIVERTED"] == 0].copy()
        
    # Drop rows where DEP_DEL15 is missing
    df = df.dropna(subset=["DEP_DEL15"]).copy()
    df["DEP_DEL15"] = df["DEP_DEL15"].astype(int)
    
    # Ensure DAY_OF_MONTH and DAY_OF_WEEK are integers
    df["DAY_OF_MONTH"] = df["DAY_OF_MONTH"].astype(int)
    df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype(int) # 1=Mon, 7=Sun in BTS
    
    print(f"Cleaned valid flight records: {len(df):,}")
    print(f"Target distribution:\n{df['DEP_DEL15'].value_counts(normalize=True)}")
    return df

def extract_leakage_free_features(df):
    """
    Construct strictly pre-departure features.
    Zero usage of realized actual times (DEP_TIME, ARR_TIME, WHEELS_OFF/ON, TAXI, DELAY breakdown).
    """
    print("\nEngineering features...")
    
    # 1. Temporal Features
    # CRS_DEP_TIME is in format HHMM (e.g. 830 for 08:30, 1545 for 15:45)
    dep_hour = (df["CRS_DEP_TIME"] // 100).clip(0, 23).astype(int)
    dep_min = (df["CRS_DEP_TIME"] % 100).clip(0, 59).astype(int)
    df["sched_dep_hour"] = dep_hour
    df["sched_dep_minute"] = dep_min
    df["sched_dep_time_min"] = dep_hour * 60 + dep_min
    
    arr_hour = (df["CRS_ARR_TIME"] // 100).clip(0, 23).astype(int)
    arr_min = (df["CRS_ARR_TIME"] % 100).clip(0, 59).astype(int)
    df["sched_arr_hour"] = arr_hour
    df["sched_arr_minute"] = arr_min
    df["sched_arr_time_min"] = arr_hour * 60 + arr_min
    
    # Cyclical hour encoding
    df["dep_hour_sin"] = np.sin(2 * np.pi * df["sched_dep_hour"] / 24.0)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["sched_dep_hour"] / 24.0)
    
    # Day of week cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7.0)
    df["is_weekend"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    
    # Time of day buckets (Aviation operational windows)
    # Early morning (00:00-06:00), Morning rush (07:00-10:00), Midday (11:00-15:00), Evening rush (16:00-20:00), Late night (21:00-23:00)
    df["time_bucket"] = pd.cut(
        df["sched_dep_hour"],
        bins=[-1, 6, 10, 15, 20, 24],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Flight duration & speed proxies
    df["sched_elapsed_time"] = df["CRS_ELAPSED_TIME"].fillna(df["DISTANCE"] / 7.0 + 30)
    df["distance"] = df["DISTANCE"].fillna(df["DISTANCE"].median())
    df["sched_speed_proxy"] = df["distance"] / (df["sched_elapsed_time"] + 1e-5)
    
    # Distance categories
    df["is_short_haul"] = (df["distance"] < 500).astype(int)
    df["is_long_haul"] = (df["distance"] > 1500).astype(int)
    
    # 2. Aircraft Continuity & Turnaround Propagation Features
    # Sort by TAIL_NUM, DAY_OF_MONTH, sched_dep_time_min
    print("Computing aircraft continuity and propagation features...")
    df["TAIL_NUM"] = df["TAIL_NUM"].fillna("UNKNOWN")
    df = df.sort_values(["TAIL_NUM", "DAY_OF_MONTH", "sched_dep_time_min"]).reset_index(drop=True)
    
    # Identify same-aircraft previous flight on the same day
    df["same_tail"] = (df["TAIL_NUM"] == df["TAIL_NUM"].shift(1)) & (df["TAIL_NUM"] != "UNKNOWN")
    df["same_day"] = df["DAY_OF_MONTH"] == df["DAY_OF_MONTH"].shift(1)
    df["valid_prev_leg"] = df["same_tail"] & df["same_day"]
    
    # Previous scheduled arrival time in minutes from midnight
    prev_arr_min = df["sched_arr_time_min"].shift(1)
    
    # Scheduled turnaround buffer: current CRS_DEP_TIME - prev CRS_ARR_TIME
    turnaround_buffer = np.where(df["valid_prev_leg"], df["sched_dep_time_min"] - prev_arr_min, 999.0)
    df["turnaround_buffer_min"] = turnaround_buffer
    df["is_tight_turnaround"] = ((df["turnaround_buffer_min"] >= 0) & (df["turnaround_buffer_min"] < 45)).astype(int)
    df["is_negative_buffer"] = (df["turnaround_buffer_min"] < 0).astype(int)
    
    # Number of flights operated by this tail earlier today
    df["tail_flight_seq_today"] = df.groupby(["TAIL_NUM", "DAY_OF_MONTH"]).cumcount()
    
    # Previous leg origin and destination matching
    prev_dest = df["DEST"].shift(1)
    df["tail_prev_dest_matches_origin"] = np.where(df["valid_prev_leg"], (prev_dest == df["ORIGIN"]).astype(int), 1)
    
    # Restore chronological order
    df = df.sort_values(["DAY_OF_MONTH", "sched_dep_time_min"]).reset_index(drop=True)
    
    # 3. Airport Schedule Congestion Proxies (Calculated strictly from scheduled CRS_DEP_TIME)
    print("Computing origin and destination congestion proxies...")
    origin_hour_counts = df.groupby(["DAY_OF_MONTH", "ORIGIN", "sched_dep_hour"])["OP_CARRIER_FL_NUM"].transform("count")
    df["origin_hourly_sched_flights"] = origin_hour_counts
    
    df["origin_daily_sched_flights"] = df.groupby(["DAY_OF_MONTH", "ORIGIN"])["OP_CARRIER_FL_NUM"].transform("count")
    df["dest_daily_sched_flights"] = df.groupby(["DAY_OF_MONTH", "DEST"])["OP_CARRIER_FL_NUM"].transform("count")
    
    df["carrier_origin_flights_today"] = df.groupby(["DAY_OF_MONTH", "ORIGIN", "OP_UNIQUE_CARRIER"])["OP_CARRIER_FL_NUM"].transform("count")
    df["carrier_origin_market_share"] = df["carrier_origin_flights_today"] / (df["origin_daily_sched_flights"] + 1e-5)
    
    # Route representation
    df["route"] = df["ORIGIN"] + "_" + df["DEST"]
    
    # Clean up temporary helper columns
    df = df.drop(columns=["same_tail", "same_day", "valid_prev_leg"])
    return df

def compute_historical_target_encodings(train_df, val_df, test_df):
    """
    Compute out-of-fold historical delay rates strictly on the Training set,
    and map smoothly to Val and Test with Bayesian smoothing / Laplace prior.
    """
    print("\nComputing Bayesian-smoothed historical delay rates from training partition...")
    global_mean = train_df["DEP_DEL15"].mean()
    m_weight = 10.0 # Smoothing weight (prior count)
    
    encoding_cols = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "route",
        ("ORIGIN", "sched_dep_hour"),
        ("OP_UNIQUE_CARRIER", "ORIGIN"),
        ("OP_UNIQUE_CARRIER", "sched_dep_hour")
    ]
    
    stats_dict = {"global_mean": float(global_mean)}
    
    for item in encoding_cols:
        if isinstance(item, tuple):
            col_name = "_x_".join(item)
            keys = list(item)
        else:
            col_name = item
            keys = [item]
            
        enc_name = f"hist_delay_rate_{col_name}"
        
        # Groupby on train
        agg = train_df.groupby(keys)["DEP_DEL15"].agg(["count", "mean"]).reset_index()
        # Bayesian smoothed rate: (count * mean + m * global_mean) / (count + m)
        agg[enc_name] = (agg["count"] * agg["mean"] + m_weight * global_mean) / (agg["count"] + m_weight)
        
        # Save mapping
        stats_dict[enc_name] = agg[keys + [enc_name]].to_dict(orient="records")
        
        # Merge back to train, val, test
        train_df = train_df.merge(agg[keys + [enc_name]], on=keys, how="left")
        train_df[enc_name] = train_df[enc_name].fillna(global_mean)
        
        val_df = val_df.merge(agg[keys + [enc_name]], on=keys, how="left")
        val_df[enc_name] = val_df[enc_name].fillna(global_mean)
        
        test_df = test_df.merge(agg[keys + [enc_name]], on=keys, how="left")
        test_df[enc_name] = test_df[enc_name].fillna(global_mean)
        
    with open(os.path.join(MODEL_DIR, "historical_stats.json"), "w") as f:
        json.dump(stats_dict, f)
        
    print(f"Historical statistics computed and saved to {MODEL_DIR}/historical_stats.json")
    return train_df, val_df, test_df

if __name__ == "__main__":
    df = load_and_clean_raw_data()
    df = extract_leakage_free_features(df)
    
    # Chronological Split
    train_mask = df["DAY_OF_MONTH"] <= 21
    val_mask = (df["DAY_OF_MONTH"] >= 22) & (df["DAY_OF_MONTH"] <= 26)
    test_mask = df["DAY_OF_MONTH"] >= 27
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\n--- Chronological Split Sizes ---")
    print(f"Train Set (Days 1-21):  {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%) | Del Rate: {train_df['DEP_DEL15'].mean()*100:.2f}%")
    print(f"Val Set   (Days 22-26): {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%) | Del Rate: {val_df['DEP_DEL15'].mean()*100:.2f}%")
    print(f"Test Set  (Days 27-31): {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%) | Del Rate: {test_df['DEP_DEL15'].mean()*100:.2f}%")
    
    train_df, val_df, test_df = compute_historical_target_encodings(train_df, val_df, test_df)
    
    # Save processed partitions for rapid model training
    print("Saving processed datasets to parquet...")
    train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)
    print("Data processing pipeline complete!")
