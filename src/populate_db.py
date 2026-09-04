import os
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fdpis_flights.db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def populate_database():
    print("Loading test and validation slices into SQLite database...", flush=True)
    
    val_df = pd.read_parquet(os.path.join(RESULTS_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(RESULTS_DIR, "test.parquet"))
    
    # Combined operational dataset (Days 22 to 31)
    df = pd.concat([val_df, test_df], ignore_index=True)
    print(f"Total operational pool: {len(df):,} flights", flush=True)
    
    with open(os.path.join(MODELS_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        df[col] = df[col].fillna(0)
        
    X = df[features].values
    
    print("Computing ensemble classification probabilities...", flush=True)
    lgb_clf = joblib.load(os.path.join(MODELS_DIR, "lightgbm.pkl"))
    xgb_clf = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    cb_clf = joblib.load(os.path.join(MODELS_DIR, "catboost.pkl"))
    
    p_lgb = lgb_clf.predict_proba(X)[:, 1]
    p_xgb = xgb_clf.predict_proba(X)[:, 1]
    p_cb = cb_clf.predict_proba(X)[:, 1]
    probs = p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30
    df["delay_prob"] = np.round(probs, 4)
    
    print("Computing regression delay predictions...", flush=True)
    regressor = joblib.load(os.path.join(MODELS_DIR, "delay_regressor.pkl"))
    pred_delay_log = regressor.predict(X)
    df["predicted_delay"] = np.round(np.clip(np.expm1(pred_delay_log), 0, None), 1)
    
    # Categorize Risk
    # High Risk: >= 0.40 probability (or delay >= 45m)
    # Elevated Risk: 0.20 to 0.40 probability
    # Normal: < 0.20 probability
    conditions = [
        (df["delay_prob"] >= 0.40),
        (df["delay_prob"] >= 0.20) & (df["delay_prob"] < 0.40),
        (df["delay_prob"] < 0.20)
    ]
    choices = ["HIGH", "ELEVATED", "NORMAL"]
    df["risk_category"] = np.select(conditions, choices, default="NORMAL")
    
    # Create clean deterministic flight_id
    df["flight_id"] = (
        df["OP_UNIQUE_CARRIER"] + "_" + 
        df["ORIGIN"] + "_" + 
        df["DEST"] + "_" + 
        df["DAY_OF_MONTH"].astype(str) + "_" + 
        df["CRS_DEP_TIME"].astype(str)
    )
    
    # Handle duplicates by taking the first occurrence
    df = df.drop_duplicates(subset=["flight_id"]).copy()
    
    # Select columns for SQLite insertion
    db_records = pd.DataFrame({
        "flight_id": df["flight_id"],
        "carrier": df["OP_UNIQUE_CARRIER"],
        "tail_num": df["TAIL_NUM"],
        "origin": df["ORIGIN"],
        "dest": df["DEST"],
        "day_of_month": df["DAY_OF_MONTH"].astype(int),
        "day_of_week": df["DAY_OF_WEEK"].astype(int),
        "crs_dep_time": df["CRS_DEP_TIME"].astype(int),
        "crs_arr_time": df["CRS_ARR_TIME"].astype(int),
        "distance": df["DISTANCE"].astype(float),
        "sched_elapsed_time": df["CRS_ELAPSED_TIME"].fillna(0).astype(float),
        "delay_prob": df["delay_prob"].astype(float),
        "predicted_delay": df["predicted_delay"].astype(float),
        "risk_category": df["risk_category"],
        "is_delayed_actual": df["DEP_DEL15"].astype(int),
        "dep_delay_actual": df["DEP_DELAY"].fillna(0).astype(float)
    })
    
    print(f"Inserting {len(db_records):,} verified flight records into SQLite database...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    db_records.to_sql("flights", conn, if_exists="replace", index=False)
    
    # Re-create indexes
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_carrier ON flights(carrier);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_origin ON flights(origin);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dest ON flights(dest);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tail ON flights(tail_num);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_day ON flights(day_of_month);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk ON flights(risk_category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prob ON flights(delay_prob);")
    conn.commit()
    conn.close()
    
    print(f"SQLite operational database populated at {DB_PATH} successfully!")

if __name__ == "__main__":
    populate_database()
