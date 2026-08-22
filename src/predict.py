import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = r"C:\Users\ashis\.gemini\antigravity\scratch\fdpis_ml\models"

class FDPISPredictor:
    """
    Self-contained inference predictor for FDPIS delay classification.
    Takes raw flight schedule parameters and outputs prediction.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        
        # Load feature definitions
        with open(os.path.join(model_dir, "features.json"), "r") as f:
            self.features = json.load(f)
            
        # Load historical statistics mapping
        with open(os.path.join(model_dir, "historical_stats.json"), "r") as f:
            self.historical_stats = json.load(f)
            
        # Load winning model & metadata
        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.optimal_threshold = self.metadata.get("optimal_threshold", 0.50)
        self.model_name = self.metadata.get("best_model_name", "ensemble")
        
        # Load model objects
        if "Ensemble" in self.model_name:
            self.lgb = joblib.load(os.path.join(model_dir, "lightgbm.pkl"))
            self.xgb = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
            self.cb = joblib.load(os.path.join(model_dir, "catboost.pkl"))
        elif "LightGBM" in self.model_name:
            self.lgb = joblib.load(os.path.join(model_dir, "lightgbm.pkl"))
        elif "XGBoost" in self.model_name:
            self.xgb = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
        elif "CatBoost" in self.model_name:
            self.cb = joblib.load(os.path.join(model_dir, "catboost.pkl"))

    def preprocess(self, df):
        df = df.copy()
        
        # Parse times
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
        
        df["dep_hour_sin"] = np.sin(2 * np.pi * df["sched_dep_hour"] / 24.0)
        df["dep_hour_cos"] = np.cos(2 * np.pi * df["sched_dep_hour"] / 24.0)
        
        df["dow_sin"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7.0)
        df["is_weekend"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
        
        df["time_bucket"] = pd.cut(
            df["sched_dep_hour"],
            bins=[-1, 6, 10, 15, 20, 24],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        df["sched_elapsed_time"] = df["CRS_ELAPSED_TIME"].fillna(df["DISTANCE"] / 7.0 + 30)
        df["distance"] = df["DISTANCE"].fillna(df["DISTANCE"].median() if "DISTANCE" in df else 800)
        df["sched_speed_proxy"] = df["distance"] / (df["sched_elapsed_time"] + 1e-5)
        
        df["is_short_haul"] = (df["distance"] < 500).astype(int)
        df["is_long_haul"] = (df["distance"] > 1500).astype(int)
        
        # Turnaround features
        if "turnaround_buffer_min" not in df.columns:
            df["turnaround_buffer_min"] = 999.0
            df["is_tight_turnaround"] = 0
            df["is_negative_buffer"] = 0
            df["tail_flight_seq_today"] = 0
            df["tail_prev_dest_matches_origin"] = 1
            
        # Congestion defaults if missing
        if "origin_hourly_sched_flights" not in df.columns:
            df["origin_hourly_sched_flights"] = 15.0
            df["origin_daily_sched_flights"] = 150.0
            df["dest_daily_sched_flights"] = 150.0
            df["carrier_origin_flights_today"] = 30.0
            df["carrier_origin_market_share"] = 0.20
            
        df["route"] = df["ORIGIN"] + "_" + df["DEST"]
        
        # Map historical encodings
        global_mean = self.historical_stats.get("global_mean", 0.1912)
        
        encoding_keys = [
            ("hist_delay_rate_OP_UNIQUE_CARRIER", ["OP_UNIQUE_CARRIER"]),
            ("hist_delay_rate_ORIGIN", ["ORIGIN"]),
            ("hist_delay_rate_DEST", ["DEST"]),
            ("hist_delay_rate_route", ["route"]),
            ("hist_delay_rate_ORIGIN_x_sched_dep_hour", ["ORIGIN", "sched_dep_hour"]),
            ("hist_delay_rate_OP_UNIQUE_CARRIER_x_ORIGIN", ["OP_UNIQUE_CARRIER", "ORIGIN"]),
            ("hist_delay_rate_OP_UNIQUE_CARRIER_x_sched_dep_hour", ["OP_UNIQUE_CARRIER", "sched_dep_hour"])
        ]
        
        for enc_name, keys in encoding_keys:
            if enc_name in self.historical_stats:
                mapping_table = pd.DataFrame(self.historical_stats[enc_name])
                df = df.merge(mapping_table, on=keys, how="left")
                df[enc_name] = df[enc_name].fillna(global_mean)
            else:
                df[enc_name] = global_mean
                
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
            
        return df[self.features].values

    def predict_proba(self, df):
        X = self.preprocess(df)
        if "Ensemble" in self.model_name:
            p_lgb = self.lgb.predict_proba(X)[:, 1]
            p_xgb = self.xgb.predict_proba(X)[:, 1]
            p_cb = self.cb.predict_proba(X)[:, 1]
            return p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30
        elif "LightGBM" in self.model_name:
            return self.lgb.predict_proba(X)[:, 1]
        elif "XGBoost" in self.model_name:
            return self.xgb.predict_proba(X)[:, 1]
        elif "CatBoost" in self.model_name:
            return self.cb.predict_proba(X)[:, 1]
        else:
            return self.lgb.predict_proba(X)[:, 1]

    def predict(self, df):
        probs = self.predict_proba(df)
        return (probs >= self.optimal_threshold).astype(int)

if __name__ == "__main__":
    print("FDPIS Predictor Class ready.")
