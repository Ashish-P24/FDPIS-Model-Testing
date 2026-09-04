import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def run_regression_suite():
    print("Loading train, val, and test splits for delay regression...", flush=True)
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)
        
    for col in features:
        train_df[col] = train_df[col].fillna(0)
        val_df[col] = val_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
        
    # We evaluate regression on two legitimate populations:
    # 1. ALL FLIGHTS (unconditional expected delay magnitude in minutes, clipped at 0 for early departures)
    # 2. DELAY-CONDITIONAL (delay magnitude among flights with DEP_DEL15 == 1)
    
    # Target: DEP_DELAY (clip negative early departures to 0 for operational delay magnitude)
    y_train_raw = train_df["DEP_DELAY"].values
    y_val_raw = val_df["DEP_DELAY"].values
    y_test_raw = test_df["DEP_DELAY"].values
    
    # Clip negative values (e.g. -15 min departure is 0 min delay)
    y_train = np.clip(y_train_raw, 0, None)
    y_val = np.clip(y_val_raw, 0, None)
    y_test = np.clip(y_test_raw, 0, None)
    
    # Log-transform target for robust training against extreme flight outliers (>1000 mins)
    # y_log = log1p(y)
    y_train_log = np.log1p(y_train)
    
    X_train = train_df[features].values
    X_val = val_df[features].values
    X_test = test_df[features].values
    
    print(f"Train rows: {len(y_train):,} | Val rows: {len(y_val):,} | Test rows: {len(y_test):,}")
    
    # Baseline 1: Mean baseline
    mean_val = np.mean(y_train)
    val_mean_mae = mean_absolute_error(y_val, np.full_like(y_val, mean_val))
    val_mean_rmse = np.sqrt(mean_squared_error(y_val, np.full_like(y_val, mean_val)))
    
    # Baseline 2: Median baseline
    med_val = np.median(y_train)
    val_med_mae = mean_absolute_error(y_val, np.full_like(y_val, med_val))
    val_med_rmse = np.sqrt(mean_squared_error(y_val, np.full_like(y_val, med_val)))
    
    print(f"\n--- Baselines on Validation Set ---")
    print(f"Mean Baseline ({mean_val:.2f} min):   MAE = {val_mean_mae:.2f} min, RMSE = {val_mean_rmse:.2f} min")
    print(f"Median Baseline ({med_val:.2f} min): MAE = {val_med_mae:.2f} min, RMSE = {val_med_rmse:.2f} min")
    
    results = [
        {"model": "Mean Baseline", "mae": val_mean_mae, "rmse": val_mean_rmse, "r2": 0.0, "med_ae": median_absolute_error(y_val, np.full_like(y_val, mean_val))},
        {"model": "Median Baseline", "mae": val_med_mae, "rmse": val_med_rmse, "r2": r2_score(y_val, np.full_like(y_val, med_val)), "med_ae": 0.0}
    ]
    
    # Model 1: Ridge Regression
    print("\nTraining Ridge Regressor...", flush=True)
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train, y_train_log)
    val_pred_ridge = np.expm1(ridge.predict(X_val))
    val_pred_ridge = np.clip(val_pred_ridge, 0, None)
    mae_ridge = mean_absolute_error(y_val, val_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_val, val_pred_ridge))
    r2_ridge = r2_score(y_val, val_pred_ridge)
    med_ridge = median_absolute_error(y_val, val_pred_ridge)
    results.append({"model": "Ridge Regression", "mae": mae_ridge, "rmse": rmse_ridge, "r2": r2_ridge, "med_ae": med_ridge})
    print(f"Ridge: MAE={mae_ridge:.2f} min, RMSE={rmse_ridge:.2f} min, R2={r2_ridge:.4f}")
    
    # Model 2: LightGBM Regressor (Huber Loss for outlier resilience)
    print("\nTraining LightGBM Regressor (Huber objective)...", flush=True)
    lgb_reg = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_reg.fit(X_train, y_train_log)
    val_pred_lgb = np.expm1(lgb_reg.predict(X_val))
    val_pred_lgb = np.clip(val_pred_lgb, 0, None)
    mae_lgb = mean_absolute_error(y_val, val_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_val, val_pred_lgb))
    r2_lgb = r2_score(y_val, val_pred_lgb)
    med_lgb = median_absolute_error(y_val, val_pred_lgb)
    results.append({"model": "LightGBM Regressor", "mae": mae_lgb, "rmse": rmse_lgb, "r2": r2_lgb, "med_ae": med_lgb})
    print(f"LightGBM Regressor: MAE={mae_lgb:.2f} min, RMSE={rmse_lgb:.2f} min, R2={r2_lgb:.4f}")
    
    # Model 3: CatBoost Regressor
    print("\nTraining CatBoost Regressor...", flush=True)
    cb_reg = CatBoostRegressor(
        iterations=300,
        learning_rate=0.06,
        depth=7,
        loss_function="MAE",
        random_seed=42,
        verbose=100,
        thread_count=-1
    )
    cb_reg.fit(X_train, y_train_log)
    val_pred_cb = np.expm1(cb_reg.predict(X_val))
    val_pred_cb = np.clip(val_pred_cb, 0, None)
    mae_cb = mean_absolute_error(y_val, val_pred_cb)
    rmse_cb = np.sqrt(mean_squared_error(y_val, val_pred_cb))
    r2_cb = r2_score(y_val, val_pred_cb)
    med_cb = median_absolute_error(y_val, val_pred_cb)
    results.append({"model": "CatBoost Regressor", "mae": mae_cb, "rmse": rmse_cb, "r2": r2_cb, "med_ae": med_cb})
    print(f"CatBoost Regressor: MAE={mae_cb:.2f} min, RMSE={rmse_cb:.2f} min, R2={r2_cb:.4f}")
    
    # Save validation comparison
    reg_df = pd.DataFrame(results).sort_values("mae")
    reg_df.to_csv(os.path.join(DATA_DIR, "regression_comparison.csv"), index=False)
    print("\n--- Regression Model Comparison ---")
    print(reg_df.to_string(index=False))
    
    # Select winning regressor and evaluate on locked test set
    best_reg_model = lgb_reg if mae_lgb <= mae_cb else cb_reg
    best_name = "LightGBM Regressor" if mae_lgb <= mae_cb else "CatBoost Regressor"
    
    print(f"\nEvaluating Best Regressor ({best_name}) on LOCKED TEST SET...", flush=True)
    test_pred = np.expm1(best_reg_model.predict(X_test))
    test_pred = np.clip(test_pred, 0, None)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    test_med_ae = median_absolute_error(y_test, test_pred)
    
    test_results = {
        "regressor_model": best_name,
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "test_median_ae": float(test_med_ae),
        "test_sample_size": len(y_test)
    }
    
    print(f"Locked Test Set -> MAE: {test_mae:.2f} min | RMSE: {test_rmse:.2f} min | R²: {test_r2:.4f} | MedAE: {test_med_ae:.2f} min")
    
    with open(os.path.join(DATA_DIR, "final_test_regression.json"), "w") as f:
        json.dump(test_results, f, indent=2)
        
    # Save the trained regressor binary
    joblib.dump(best_reg_model, os.path.join(MODEL_DIR, "delay_regressor.pkl"))
    print(f"Saved best regressor to {MODEL_DIR}/delay_regressor.pkl")

if __name__ == "__main__":
    run_regression_suite()
