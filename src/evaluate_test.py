import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

DATA_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"
MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

def run_final_test_evaluation():
    print("================ FINAL LOCKED TEST SET EVALUATION ================", flush=True)
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        test_df[col] = test_df[col].fillna(0)
        
    X_test = test_df[features].values
    y_test = test_df["DEP_DEL15"].values
    
    print(f"Test Set Size: {len(test_df):,} flights")
    print(f"Test Class 0 (On-time): {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"Test Class 1 (Delayed): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")
    
    # Load Models
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    cb_model = joblib.load(os.path.join(MODEL_DIR, "catboost.pkl"))
    
    # Load optimal threshold from validation set
    with open(os.path.join(DATA_DIR, "model_comparison.csv"), "r") as f:
        comp_df = pd.read_csv(f)
        best_row = comp_df.iloc[0]
        best_model_name = best_row["model"]
        best_threshold = float(best_row["threshold"])
        val_acc = float(best_row["accuracy"])
        
    print(f"\nWinning Model Architecture: {best_model_name}")
    print(f"Optimal Classification Threshold (from Validation Set): {best_threshold:.2f}")
    print(f"Best Validation Accuracy: {val_acc*100:.2f}%")
    
    # Test predictions
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    p_cb = cb_model.predict_proba(X_test)[:, 1]
    
    if "Ensemble" in best_model_name:
        test_probs = p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30
    elif "LightGBM" in best_model_name:
        test_probs = p_lgb
    elif "XGBoost" in best_model_name:
        test_probs = p_xgb
    elif "CatBoost" in best_model_name:
        test_probs = p_cb
    else:
        test_probs = p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30
        
    test_preds = (test_probs >= best_threshold).astype(int)
    
    # Final Metrics
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_auc = roc_auc_score(y_test, test_probs)
    test_prauc = average_precision_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds).tolist()
    
    test_metrics = {
        "best_model_name": best_model_name,
        "best_val_accuracy": val_acc,
        "final_test_accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1),
        "roc_auc": float(test_auc),
        "pr_auc": float(test_prauc),
        "confusion_matrix": cm,
        "optimal_threshold": float(best_threshold),
        "test_set_size": int(len(test_df)),
        "class_0_count": int((y_test == 0).sum()),
        "class_1_count": int((y_test == 1).sum())
    }
    
    # Save metadata for predictor
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    with open(os.path.join(DATA_DIR, "final_test_evaluation.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    print("\n---------------- FINAL TEST PERFORMANCE RESULTS ----------------")
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"Precision:           {test_prec*100:.2f}%")
    print(f"Recall:              {test_rec*100:.2f}%")
    print(f"F1-Score:            {test_f1*100:.2f}%")
    print(f"ROC-AUC:             {test_auc:.4f}")
    print(f"PR-AUC:              {test_prauc:.4f}")
    print(f"Confusion Matrix:    {cm}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_final_test_evaluation()
