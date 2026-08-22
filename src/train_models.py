import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    preds = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "threshold": float(threshold)
    }

def run_experiment_suite():
    print("Loading engineered train and val splits...", flush=True)
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    
    # Feature columns specification
    numeric_features = [
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
    
    target_col = "DEP_DEL15"
    
    # Save feature list for documentation and reproducibility
    with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
        json.dump(numeric_features, f, indent=2)
    print(f"Total features: {len(numeric_features)}", flush=True)
    
    # Fill NAs
    for col in numeric_features:
        train_df[col] = train_df[col].fillna(0)
        val_df[col] = val_df[col].fillna(0)
        
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    X_train = train_df[numeric_features].values
    X_val = val_df[numeric_features].values
    
    results = []
    val_predictions = {}
    
    # 1. Majority Class Baseline
    print("\n--- 1. Baseline: Majority Class ---", flush=True)
    t0 = time.time()
    val_probs_maj = np.full(len(y_val), float(y_train.mean()))
    maj_eval = evaluate_predictions(y_val, val_probs_maj, threshold=0.5)
    maj_eval["model"] = "Majority Class Baseline"
    results.append(maj_eval)
    val_predictions["Majority_Class"] = val_probs_maj
    print(f"Majority Class Baseline Val Acc: {maj_eval['accuracy']*100:.2f}% (time: {time.time()-t0:.1f}s)", flush=True)
    
    # 2. Logistic Regression (Standard Scaled)
    print("\n--- 2. Baseline: Logistic Regression ---", flush=True)
    t0 = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    lr = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", random_state=42)
    lr.fit(X_train_scaled, y_train)
    val_probs_lr = lr.predict_proba(X_val_scaled)[:, 1]
    best_t_lr, best_acc_lr = find_optimal_threshold(y_val, val_probs_lr)
    lr_eval = evaluate_predictions(y_val, val_probs_lr, threshold=best_t_lr)
    lr_eval["model"] = "Logistic Regression"
    results.append(lr_eval)
    val_predictions["LogisticRegression"] = val_probs_lr
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    print(f"Logistic Regression Val Acc: {lr_eval['accuracy']*100:.2f}% (thresh={best_t_lr:.2f}, AUC={lr_eval['roc_auc']:.4f}, time: {time.time()-t0:.1f}s)", flush=True)
    
    # 3. HistGradientBoostingClassifier
    print("\n--- 3. HistGradientBoostingClassifier ---", flush=True)
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42)
    hgb.fit(X_train, y_train)
    val_probs_hgb = hgb.predict_proba(X_val)[:, 1]
    best_t_hgb, _ = find_optimal_threshold(y_val, val_probs_hgb)
    hgb_eval = evaluate_predictions(y_val, val_probs_hgb, threshold=best_t_hgb)
    hgb_eval["model"] = "HistGradientBoosting"
    results.append(hgb_eval)
    val_predictions["HistGradientBoosting"] = val_probs_hgb
    joblib.dump(hgb, os.path.join(MODEL_DIR, "hist_gradient_boosting.pkl"))
    print(f"HistGradientBoosting Val Acc: {hgb_eval['accuracy']*100:.2f}% (thresh={best_t_hgb:.2f}, AUC={hgb_eval['roc_auc']:.4f}, time: {time.time()-t0:.1f}s)", flush=True)
    
    # 4. LightGBM
    print("\n--- 4. LightGBM Classifier ---", flush=True)
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    val_probs_lgb = lgb_model.predict_proba(X_val)[:, 1]
    best_t_lgb, _ = find_optimal_threshold(y_val, val_probs_lgb)
    lgb_eval = evaluate_predictions(y_val, val_probs_lgb, threshold=best_t_lgb)
    lgb_eval["model"] = "LightGBM"
    results.append(lgb_eval)
    val_predictions["LightGBM"] = val_probs_lgb
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lightgbm.pkl"))
    print(f"LightGBM Val Acc: {lgb_eval['accuracy']*100:.2f}% (thresh={best_t_lgb:.2f}, AUC={lgb_eval['roc_auc']:.4f}, time: {time.time()-t0:.1f}s)", flush=True)
    
    # 5. XGBoost Classifier
    print("\n--- 5. XGBoost Classifier ---", flush=True)
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    xgb_model.fit(X_train, y_train)
    val_probs_xgb = xgb_model.predict_proba(X_val)[:, 1]
    best_t_xgb, _ = find_optimal_threshold(y_val, val_probs_xgb)
    xgb_eval = evaluate_predictions(y_val, val_probs_xgb, threshold=best_t_xgb)
    xgb_eval["model"] = "XGBoost"
    results.append(xgb_eval)
    val_predictions["XGBoost"] = val_probs_xgb
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost.pkl"))
    print(f"XGBoost Val Acc: {xgb_eval['accuracy']*100:.2f}% (thresh={best_t_xgb:.2f}, AUC={xgb_eval['roc_auc']:.4f}, time: {time.time()-t0:.1f}s)", flush=True)
    
    # 6. CatBoost Classifier
    print("\n--- 6. CatBoost Classifier ---", flush=True)
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=7,
        eval_metric="Logloss",
        random_seed=42,
        verbose=100,
        thread_count=-1
    )
    cb_model.fit(X_train, y_train)
    val_probs_cb = cb_model.predict_proba(X_val)[:, 1]
    best_t_cb, _ = find_optimal_threshold(y_val, val_probs_cb)
    cb_eval = evaluate_predictions(y_val, val_probs_cb, threshold=best_t_cb)
    cb_eval["model"] = "CatBoost"
    results.append(cb_eval)
    val_predictions["CatBoost"] = val_probs_cb
    joblib.dump(cb_model, os.path.join(MODEL_DIR, "catboost.pkl"))
    print(f"CatBoost Val Acc: {cb_eval['accuracy']*100:.2f}% (thresh={best_t_cb:.2f}, AUC={cb_eval['roc_auc']:.4f}, time: {time.time()-t0:.1f}s)", flush=True)
    
    # 7. Blended Ensemble (LightGBM + XGBoost + CatBoost)
    print("\n--- 7. Blended Ensemble (LGBM + XGB + CatBoost) ---", flush=True)
    val_probs_blend = (val_probs_lgb * 0.35 + val_probs_xgb * 0.35 + val_probs_cb * 0.30)
    best_t_blend, _ = find_optimal_threshold(y_val, val_probs_blend)
    blend_eval = evaluate_predictions(y_val, val_probs_blend, threshold=best_t_blend)
    blend_eval["model"] = "Ensemble Blend (LGBM + XGB + CatBoost)"
    results.append(blend_eval)
    val_predictions["Blend_LGB_XGB_CB"] = val_probs_blend
    print(f"Ensemble Blend Val Acc: {blend_eval['accuracy']*100:.2f}% (thresh={best_t_blend:.2f}, AUC={blend_eval['roc_auc']:.4f})", flush=True)
    
    # Feature Importance (from LightGBM and XGBoost)
    feat_imp = pd.DataFrame({
        "feature": numeric_features,
        "lgb_importance": lgb_model.feature_importances_,
        "xgb_importance": xgb_model.feature_importances_
    }).sort_values("lgb_importance", ascending=False)
    feat_imp.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 Most Important Predictive Features:", flush=True)
    print(feat_imp.head(10).to_string(index=False), flush=True)
    
    # Save comparison table
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("accuracy", ascending=False)
    res_df.to_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index=False)
    print("\n================ FINAL COMPARISON TABLE ================", flush=True)
    print(res_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]].to_string(index=False), flush=True)

if __name__ == "__main__":
    run_experiment_suite()
