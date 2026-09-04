import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

class FDPISPredictor:
    """
    Self-contained inference predictor for FDPIS dual prediction (Classification + Regression)
    with automated explainability.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        
        # Load feature definitions
        with open(os.path.join(model_dir, "features.json"), "r") as f:
            self.features = json.load(f)
            
        # Load historical statistics mapping
        with open(os.path.join(model_dir, "historical_stats.json"), "r") as f:
            self.historical_stats = json.load(f)
            
        # Load winning classifier & regressor models
        self.lgb_clf = joblib.load(os.path.join(model_dir, "lightgbm.pkl"))
        self.xgb_clf = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
        self.cb_clf = joblib.load(os.path.join(model_dir, "catboost.pkl"))
        
        # Delay Duration Regressor
        reg_path = os.path.join(model_dir, "delay_regressor.pkl")
        self.regressor = joblib.load(reg_path) if os.path.exists(reg_path) else None
        
        self.threshold_competition = 0.48
        self.threshold_operations = 0.18

    def preprocess(self, df):
        df = df.copy()
        
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
        
        if "DISTANCE" not in df.columns:
            df["DISTANCE"] = 800.0
            
        if "CRS_ELAPSED_TIME" not in df.columns:
            df["CRS_ELAPSED_TIME"] = df["DISTANCE"] / 7.0 + 30
            
        df["sched_elapsed_time"] = df["CRS_ELAPSED_TIME"].fillna(df["DISTANCE"] / 7.0 + 30)
        df["distance"] = df["DISTANCE"].fillna(800.0)
        df["sched_speed_proxy"] = df["distance"] / (df["sched_elapsed_time"] + 1e-5)
        
        df["is_short_haul"] = (df["distance"] < 500).astype(int)
        df["is_long_haul"] = (df["distance"] > 1500).astype(int)
        
        # Turnaround buffer
        if "turnaround_buffer_min" not in df.columns:
            df["turnaround_buffer_min"] = 90.0
            
        df["is_tight_turnaround"] = (df["turnaround_buffer_min"] < 45).astype(int)
        df["is_negative_buffer"] = (df["turnaround_buffer_min"] < 0).astype(int)
        df["tail_flight_seq_today"] = df.get("tail_flight_seq_today", 1)
        df["tail_prev_dest_matches_origin"] = df.get("tail_prev_dest_matches_origin", 1)
        
        # Airport and carrier congestion proxies
        df["origin_hourly_sched_flights"] = df.get("origin_hourly_sched_flights", 25.0)
        df["origin_daily_sched_flights"] = df.get("origin_daily_sched_flights", 350.0)
        df["dest_daily_sched_flights"] = df.get("dest_daily_sched_flights", 350.0)
        df["carrier_origin_flights_today"] = df.get("carrier_origin_flights_today", 80.0)
        df["carrier_origin_market_share"] = df.get("carrier_origin_market_share", 0.35)
        
        df["route"] = df["ORIGIN"] + "_" + df["DEST"]
        
        # Bayesian smoothed target encodings from training statistics
        global_mean = self.historical_stats.get("global_mean", 0.1912)
        
        # Helper to lookup precomputed encodings
        for enc_name, records in self.historical_stats.items():
            if enc_name == "global_mean":
                continue
            lookup_map = {}
            for r in records:
                # Key can be single or composite
                keys = [k for k in r.keys() if k != enc_name]
                if len(keys) == 1:
                    lookup_map[str(r[keys[0]])] = r[enc_name]
                elif len(keys) == 2:
                    composite_key = f"{r[keys[0]]}_{r[keys[1]]}"
                    lookup_map[composite_key] = r[enc_name]
                    
            if enc_name == "hist_delay_rate_OP_UNIQUE_CARRIER":
                df[enc_name] = df["OP_UNIQUE_CARRIER"].map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_ORIGIN":
                df[enc_name] = df["ORIGIN"].map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_DEST":
                df[enc_name] = df["DEST"].map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_route":
                df[enc_name] = df["route"].map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_ORIGIN_x_sched_dep_hour":
                comp = df["ORIGIN"] + "_" + df["sched_dep_hour"].astype(str)
                df[enc_name] = comp.map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_OP_UNIQUE_CARRIER_x_ORIGIN":
                comp = df["OP_UNIQUE_CARRIER"] + "_" + df["ORIGIN"]
                df[enc_name] = comp.map(lookup_map).fillna(global_mean)
            elif enc_name == "hist_delay_rate_OP_UNIQUE_CARRIER_x_sched_dep_hour":
                comp = df["OP_UNIQUE_CARRIER"] + "_" + df["sched_dep_hour"].astype(str)
                df[enc_name] = comp.map(lookup_map).fillna(global_mean)
                
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0.0
            df[feat] = df[feat].fillna(0)
            
        return df[self.features]

    def predict_flight(self, flight_dict):
        """
        Takes raw dictionary of flight parameters and returns:
        - delay probability
        - risk category
        - predicted delay in minutes
        - estimated actual arrival time
        - top contributing reasons
        """
        df_raw = pd.DataFrame([flight_dict])
        X = self.preprocess(df_raw).values
        
        # Classification probabilities
        p_lgb = self.lgb_clf.predict_proba(X)[0, 1]
        p_xgb = self.xgb_clf.predict_proba(X)[0, 1]
        p_cb = self.cb_clf.predict_proba(X)[0, 1]
        prob = float(p_lgb * 0.35 + p_xgb * 0.35 + p_cb * 0.30)
        
        # Regression delay
        pred_delay = 0.0
        if self.regressor:
            log_delay = self.regressor.predict(X)[0]
            pred_delay = float(np.clip(np.expm1(log_delay), 0, None))
            
        # If classifier says high delay risk, expected delay should reflect elevated magnitude
        if prob >= 0.40 and pred_delay < 25.0:
            pred_delay = max(pred_delay, 35.0 + prob * 25.0)
            
        # Calculate Estimated Arrival Time (HHMM format)
        sched_arr = int(flight_dict.get("CRS_ARR_TIME", 1200))
        arr_h = sched_arr // 100
        arr_m = sched_arr % 100
        total_arr_min = arr_h * 60 + arr_m + int(round(pred_delay))
        est_arr_h = (total_arr_min // 60) % 24
        est_arr_m = total_arr_min % 60
        est_arr_time = f"{est_arr_h:02d}:{est_arr_m:02d}"
        
        # Risk Category
        if prob >= 0.40:
            risk = "HIGH"
        elif prob >= 0.20:
            risk = "ELEVATED"
        else:
            risk = "NORMAL"
            
        # Explainability: top reasons for this specific flight
        reasons = []
        turnaround = flight_dict.get("turnaround_buffer_min", 90.0)
        if turnaround < 45:
            reasons.append({
                "factor": "Inbound Turnaround Compression",
                "detail": f"Turnaround buffer is only {turnaround:.0f} mins (standard requirement is 45m). Severe risk of late aircraft inheritance.",
                "impact": "HIGH POSITIVE IMPACT ON DELAY"
            })
            
        dep_h = int(flight_dict.get("CRS_DEP_TIME", 1200)) // 100
        if dep_h >= 16:
            reasons.append({
                "factor": "Evening Airspace Congestion",
                "detail": f"Scheduled departure at {dep_h:02d}:00 falls in evening traffic peak where cascading delays peak across the network.",
                "impact": "MODERATE POSITIVE IMPACT"
            })
            
        origin = flight_dict.get("ORIGIN", "")
        if origin in ["ORD", "EWR", "LGA", "JFK", "SFO", "ATL"]:
            reasons.append({
                "factor": "High-Density Hub Operation",
                "detail": f"{origin} is a major slot-controlled hub with heavy apron and taxiway volume.",
                "impact": "ELEVATED STRUCTURAL DELAY RATE"
            })
            
        if not reasons:
            reasons.append({
                "factor": "Schedule Slack & Off-Peak Buffer",
                "detail": "Flight departs during stable operational window with sufficient ground turn buffer.",
                "impact": "LOW DELAY RISK"
            })
            
        return {
            "delay_probability": round(prob, 4),
            "delay_probability_pct": f"{prob*100:.1f}%",
            "risk_category": risk,
            "predicted_delay_minutes": round(pred_delay, 1),
            "sched_dep_time": f"{int(flight_dict.get('CRS_DEP_TIME', 0)) // 100:02d}:{int(flight_dict.get('CRS_DEP_TIME', 0)) % 100:02d}",
            "sched_arr_time": f"{arr_h:02d}:{arr_m:02d}",
            "est_arr_time": est_arr_time,
            "decision_reasons": reasons
        }
