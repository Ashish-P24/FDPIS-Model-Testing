import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"
MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

def run_error_analysis():
    print("Running detailed error analysis on validation set...", flush=True)
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        val_df[col] = val_df[col].fillna(0)
        
    X_val = val_df[features].values
    y_val = val_df["DEP_DEL15"].values
    
    # Load LightGBM, XGBoost, and CatBoost models
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    cb_model = joblib.load(os.path.join(MODEL_DIR, "catboost.pkl"))
    
    # Best ensemble predictions
    val_probs = (
        lgb_model.predict_proba(X_val)[:, 1] * 0.35 +
        xgb_model.predict_proba(X_val)[:, 1] * 0.35 +
        cb_model.predict_proba(X_val)[:, 1] * 0.30
    )
    
    # Threshold that maximizes validation accuracy
    best_thresh = 0.50
    with open(os.path.join(DATA_DIR, "model_comparison.csv"), "r") as f:
        comp_df = pd.read_csv(f)
        best_row = comp_df.iloc[0]
        best_thresh = float(best_row["threshold"])
        
    val_df["pred_prob"] = val_probs
    val_df["pred_label"] = (val_probs >= best_thresh).astype(int)
    val_df["is_error"] = (val_df["pred_label"] != val_df["DEP_DEL15"]).astype(int)
    val_df["error_type"] = "CORRECT"
    val_df.loc[(val_df["pred_label"] == 1) & (val_df["DEP_DEL15"] == 0), "error_type"] = "FALSE_POSITIVE"
    val_df.loc[(val_df["pred_label"] == 0) & (val_df["DEP_DEL15"] == 1), "error_type"] = "FALSE_NEGATIVE"
    
    total_val = len(val_df)
    err_counts = val_df["error_type"].value_counts().to_dict()
    print(f"Overall Validation Accuracy: {(1 - val_df['is_error'].mean())*100:.2f}%")
    print(f"Error breakdown: {err_counts}")
    
    # 1. Error by Carrier
    carrier_err = val_df.groupby("OP_UNIQUE_CARRIER").agg(
        total_flights=("DEP_DEL15", "count"),
        delay_rate=("DEP_DEL15", "mean"),
        error_rate=("is_error", "mean"),
        accuracy=("is_error", lambda x: 1 - x.mean())
    ).sort_values("error_rate", ascending=False)
    
    # 2. Error by Departure Hour
    hour_err = val_df.groupby("sched_dep_hour").agg(
        total_flights=("DEP_DEL15", "count"),
        delay_rate=("DEP_DEL15", "mean"),
        accuracy=("is_error", lambda x: 1 - x.mean())
    )
    
    # 3. Error by Top 15 Origin Airports
    top_origins = val_df["ORIGIN"].value_counts().head(15).index
    origin_err = val_df[val_df["ORIGIN"].isin(top_origins)].groupby("ORIGIN").agg(
        total_flights=("DEP_DEL15", "count"),
        delay_rate=("DEP_DEL15", "mean"),
        accuracy=("is_error", lambda x: 1 - x.mean())
    ).sort_values("accuracy", ascending=True)
    
    # 4. Error by Day of Week
    dow_err = val_df.groupby("DAY_OF_WEEK").agg(
        total_flights=("DEP_DEL15", "count"),
        delay_rate=("DEP_DEL15", "mean"),
        accuracy=("is_error", lambda x: 1 - x.mean())
    )
    
    analysis_report = {
        "overall_val_accuracy": float(1 - val_df["is_error"].mean()),
        "threshold_used": float(best_thresh),
        "confusion_matrix": confusion_matrix(val_df["DEP_DEL15"], val_df["pred_label"]).tolist(),
        "error_distribution": err_counts,
        "carrier_analysis": carrier_err.reset_index().to_dict(orient="records"),
        "hourly_analysis": hour_err.reset_index().to_dict(orient="records"),
        "origin_analysis": origin_err.reset_index().to_dict(orient="records"),
        "dow_analysis": dow_err.reset_index().to_dict(orient="records")
    }
    
    with open(os.path.join(DATA_DIR, "error_analysis.json"), "w") as f:
        json.dump(analysis_report, f, indent=2)
        
    print("\n--- Carrier-Level Accuracy Breakdown ---")
    print(carrier_err)
    print("\n--- Top Hardest Origin Airports ---")
    print(origin_err.head(10))
    print("\nError analysis report saved to results/error_analysis.json")

if __name__ == "__main__":
    run_error_analysis()
