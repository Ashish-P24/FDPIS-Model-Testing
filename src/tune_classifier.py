import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def analyze_tradeoffs():
    print("Loading validation dataset and models...", flush=True)
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        val_df[col] = val_df[col].fillna(0)
        
    X_val = val_df[features].values
    y_val = val_df["DEP_DEL15"].values
    
    # Load models
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    cb_model = joblib.load(os.path.join(MODEL_DIR, "catboost.pkl"))
    
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    p_cb = cb_model.predict_proba(X_val)[:, 1]
    y_probs = p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30
    
    roc_auc = roc_auc_score(y_val, y_probs)
    pr_auc = average_precision_score(y_val, y_probs)
    print(f"Validation ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    
    # Sweep thresholds
    thresholds = np.linspace(0.10, 0.80, 71)
    results = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        results.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        })
        
    tradeoff_df = pd.DataFrame(results)
    tradeoff_df.to_csv(os.path.join(DATA_DIR, "threshold_tradeoff.csv"), index=False)
    
    # Find best thresholds for different objectives
    best_acc_idx = tradeoff_df["accuracy"].idxmax()
    best_f1_idx = tradeoff_df["f1"].idxmax()
    
    best_acc_row = tradeoff_df.loc[best_acc_idx]
    best_f1_row = tradeoff_df.loc[best_f1_idx]
    
    print("\n--- Profile 1: Maximum Accuracy (Competition Mode) ---")
    print(f"Threshold: {best_acc_row['threshold']:.2f}")
    print(f"Accuracy:  {best_acc_row['accuracy']*100:.2f}%")
    print(f"Precision: {best_acc_row['precision']*100:.2f}%")
    print(f"Recall:    {best_acc_row['recall']*100:.2f}%")
    print(f"F1-Score:  {best_f1_row['f1']:.4f}")
    
    print("\n--- Profile 2: Maximum F1 / Balanced Operational Risk (Operations Mode) ---")
    print(f"Threshold: {best_f1_row['threshold']:.2f}")
    print(f"Accuracy:  {best_f1_row['accuracy']*100:.2f}%")
    print(f"Precision: {best_f1_row['precision']*100:.2f}%")
    print(f"Recall:    {best_f1_row['recall']*100:.2f}%")
    print(f"F1-Score:  {best_f1_row['f1']:.4f}")
    
    # Ranking performance: Precision@K%
    sorted_indices = np.argsort(-y_probs)
    y_val_sorted = y_val[sorted_indices]
    
    ranking_k = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ranking_results = {}
    base_rate = float(np.mean(y_val))
    print(f"\nBase delay rate: {base_rate*100:.2f}%")
    print("Top-K% Operational Inspection Queue Quality:")
    for k in ranking_k:
        top_n = max(1, int(len(y_val) * (k / 100.0)))
        prec_at_k = float(np.mean(y_val_sorted[:top_n]))
        lift = prec_at_k / base_rate
        ranking_results[f"precision@{k}%"] = prec_at_k
        ranking_results[f"lift@{k}%"] = lift
        print(f"  Precision@{k:4.1f}% (top {top_n:5d} flights): {prec_at_k*100:5.2f}% (Lift: {lift:4.2f}x over random)")
        
    with open(os.path.join(DATA_DIR, "ranking_evaluation.json"), "w") as f:
        json.dump(ranking_results, f, indent=2)
        
    print("\nThreshold trade-off analysis and ranking metrics saved successfully!")

if __name__ == "__main__":
    analyze_tradeoffs()
