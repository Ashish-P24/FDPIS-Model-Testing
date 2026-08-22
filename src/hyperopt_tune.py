import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Suppress optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\results"
MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0.2, 0.8, 61)
    best_acc = 0.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=15):
    print("\n================ OPTUNA HYPERPARAMETER TUNING: LIGHTGBM ================", flush=True)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        best_thresh, best_acc = find_optimal_threshold(y_val, val_probs)
        return best_acc
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"LightGBM Best Trial Accuracy: {study.best_value*100:.2f}%", flush=True)
    print(f"LightGBM Best Hyperparameters: {study.best_params}", flush=True)
    return study.best_params, study.best_value

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=15):
    print("\n================ OPTUNA HYPERPARAMETER TUNING: XGBOOST ================", flush=True)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        best_thresh, best_acc = find_optimal_threshold(y_val, val_probs)
        return best_acc
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"XGBoost Best Trial Accuracy: {study.best_value*100:.2f}%", flush=True)
    print(f"XGBoost Best Hyperparameters: {study.best_params}", flush=True)
    return study.best_params, study.best_value

if __name__ == "__main__":
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        train_df[col] = train_df[col].fillna(0)
        val_df[col] = val_df[col].fillna(0)
        
    y_train = train_df["DEP_DEL15"].values
    y_val = val_df["DEP_DEL15"].values
    
    X_train = train_df[features].values
    X_val = val_df[features].values
    
    best_lgb_params, best_lgb_acc = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=12)
    best_xgb_params, best_xgb_acc = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=12)
    
    tuning_summary = {
        "lightgbm": {"best_accuracy": best_lgb_acc, "best_params": best_lgb_params},
        "xgboost": {"best_accuracy": best_xgb_acc, "best_params": best_xgb_params}
    }
    with open(os.path.join(MODEL_DIR, "tuned_hyperparameters.json"), "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print("\nTuned hyperparameters saved successfully to tuned_hyperparameters.json", flush=True)
