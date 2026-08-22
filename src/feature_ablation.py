import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb

DATA_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"
MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_acc = 0.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

def run_ablation():
    print("Loading engineered train and val splits for Feature Ablation Study...", flush=True)
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    
    y_train = train_df["DEP_DEL15"].values
    y_val = val_df["DEP_DEL15"].values
    
    # Define Feature Sets
    feature_sets = {
        "A. Raw Schedule Features": [
            "sched_dep_hour", "sched_dep_minute", "sched_arr_hour", "sched_arr_minute",
            "sched_elapsed_time", "distance"
        ],
        "B. + Temporal & Cyclical": [
            "sched_dep_hour", "sched_dep_minute", "sched_arr_hour", "sched_arr_minute",
            "sched_elapsed_time", "distance",
            "dep_hour_sin", "dep_hour_cos", "dow_sin", "dow_cos", "is_weekend", "time_bucket", "sched_speed_proxy"
        ],
        "C. + Route & Congestion": [
            "sched_dep_hour", "sched_dep_minute", "sched_arr_hour", "sched_arr_minute",
            "sched_elapsed_time", "distance",
            "dep_hour_sin", "dep_hour_cos", "dow_sin", "dow_cos", "is_weekend", "time_bucket", "sched_speed_proxy",
            "origin_hourly_sched_flights", "origin_daily_sched_flights", "dest_daily_sched_flights",
            "carrier_origin_flights_today", "carrier_origin_market_share", "is_short_haul", "is_long_haul"
        ],
        "D. + Aircraft Turnaround/Propagation": [
            "sched_dep_hour", "sched_dep_minute", "sched_arr_hour", "sched_arr_minute",
            "sched_elapsed_time", "distance",
            "dep_hour_sin", "dep_hour_cos", "dow_sin", "dow_cos", "is_weekend", "time_bucket", "sched_speed_proxy",
            "origin_hourly_sched_flights", "origin_daily_sched_flights", "dest_daily_sched_flights",
            "carrier_origin_flights_today", "carrier_origin_market_share", "is_short_haul", "is_long_haul",
            "turnaround_buffer_min", "is_tight_turnaround", "is_negative_buffer",
            "tail_flight_seq_today", "tail_prev_dest_matches_origin"
        ],
        "E. Full Feature Set (+ Historical Bayesian Rates)": [
            "sched_dep_hour", "sched_dep_minute", "sched_dep_time_min",
            "sched_arr_hour", "sched_arr_minute", "sched_arr_time_min",
            "dep_hour_sin", "dep_hour_cos",
            "dow_sin", "dow_cos", "is_weekend", "time_bucket",
            "sched_elapsed_time", "distance", "sched_speed_proxy",
            "is_short_haul", "is_long_haul",
            "turnaround_buffer_min", "is_tight_turnaround", "is_negative_buffer",
            "tail_flight_seq_today", "tail_prev_dest_matches_origin",
            "origin_hourly_sched_flights", "origin_daily_sched_flights",
            "dest_daily_sched_flights", "carrier_origin_flights_today",
            "carrier_origin_market_share",
            "hist_delay_rate_OP_UNIQUE_CARRIER",
            "hist_delay_rate_ORIGIN",
            "hist_delay_rate_DEST",
            "hist_delay_rate_route",
            "hist_delay_rate_ORIGIN_x_sched_dep_hour",
            "hist_delay_rate_OP_UNIQUE_CARRIER_x_ORIGIN",
            "hist_delay_rate_OP_UNIQUE_CARRIER_x_sched_dep_hour"
        ]
    }
    
    ablation_results = []
    
    for name, feats in feature_sets.items():
        print(f"\nEvaluating Feature Group: {name} (Features: {len(feats)})...", flush=True)
        for col in feats:
            train_df[col] = train_df[col].fillna(0)
            val_df[col] = val_df[col].fillna(0)
            
        X_train_sub = train_df[feats].values
        X_val_sub = val_df[feats].values
        
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_sub, y_train)
        val_probs = clf.predict_proba(X_val_sub)[:, 1]
        
        best_thresh, best_acc = find_optimal_threshold(y_val, val_probs)
        preds = (val_probs >= best_thresh).astype(int)
        auc = roc_auc_score(y_val, val_probs)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        print(f"-> Accuracy: {best_acc*100:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f} | Optimal Threshold: {best_thresh:.2f}", flush=True)
        ablation_results.append({
            "Feature_Group": name,
            "Feature_Count": len(feats),
            "Accuracy": float(best_acc),
            "ROC_AUC": float(auc),
            "F1_Score": float(f1),
            "Optimal_Threshold": float(best_thresh)
        })
        
    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv(os.path.join(DATA_DIR, "feature_ablation.csv"), index=False)
    print("\n================ FEATURE ABLATION SUMMARY ================", flush=True)
    print(abl_df.to_string(index=False), flush=True)

if __name__ == "__main__":
    run_ablation()
