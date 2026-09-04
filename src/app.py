import os
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fdpis_flights.db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

from src.propagation_engine import FDPISPropagationEngine
from src.predict import FDPISPredictor

app = FastAPI(
    title="FDPIS Operations Intelligence API",
    description="Flight Delay Propagation Intelligence System - Real-Time Predictive Risk & Delay Cascade Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and cache
PREDICTOR: Optional[FDPISPredictor] = None

class FlightInputRequest(BaseModel):
    carrier: str
    origin: str
    dest: str
    crs_dep_time: int
    crs_arr_time: int
    day_of_month: int = 25
    day_of_week: int = 4
    distance: float = 800.0
    tail_num: str = "N901DA"
    turnaround_buffer_min: float = 90.0

@app.on_event("startup")
def startup_event():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = FDPISPredictor(model_dir=MODELS_DIR)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "system": "Flight Delay Propagation Intelligence System (FDPIS)",
        "version": "2.0.0",
        "database_connected": os.path.exists(DB_PATH),
        "predictor_ready": PREDICTOR is not None
    }

@app.get("/api/overview")
def get_operations_overview():
    """
    Returns operational briefing KPIs: high risk flight count, network delay rate, top at-risk airports.
    """
    conn = get_db()
    cursor = conn.cursor()
    
    total_flights = cursor.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    high_risk_flights = cursor.execute("SELECT COUNT(*) FROM flights WHERE risk_category = 'HIGH'").fetchone()[0]
    elevated_risk_flights = cursor.execute("SELECT COUNT(*) FROM flights WHERE risk_category = 'ELEVATED'").fetchone()[0]
    normal_flights = cursor.execute("SELECT COUNT(*) FROM flights WHERE risk_category = 'NORMAL'").fetchone()[0]
    
    hub_risks = cursor.execute("""
        SELECT origin, 
               COUNT(*) as total_flights,
               SUM(CASE WHEN risk_category = 'HIGH' THEN 1 ELSE 0 END) as high_risk_count,
               AVG(delay_prob) as avg_risk_score
        FROM flights
        GROUP BY origin
        HAVING total_flights >= 20
        ORDER BY high_risk_count DESC
        LIMIT 8
    """).fetchall()
    
    tail_risks = cursor.execute("""
        SELECT tail_num, carrier, COUNT(*) as flight_count, AVG(delay_prob) as avg_tail_risk
        FROM flights
        WHERE tail_num != 'UNKNOWN'
        GROUP BY tail_num
        HAVING flight_count >= 3 AND avg_tail_risk >= 0.40
        ORDER BY avg_tail_risk DESC
        LIMIT 6
    """).fetchall()
    
    conn.close()
    
    return {
        "summary": {
            "total_flights": total_flights,
            "high_risk_count": high_risk_flights,
            "elevated_risk_count": elevated_risk_flights,
            "normal_count": normal_flights,
            "network_risk_pct": round((high_risk_flights / max(1, total_flights)) * 100, 2)
        },
        "critical_hubs": [dict(r) for r in hub_risks],
        "pressured_aircraft": [dict(r) for r in tail_risks]
    }

@app.get("/api/flights")
def search_flights(
    carrier: Optional[str] = None,
    origin: Optional[str] = None,
    dest: Optional[str] = None,
    tail_num: Optional[str] = None,
    risk_category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Search and filter operational flights database.
    """
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM flights WHERE 1=1"
    params = []
    
    if carrier:
        query += " AND carrier = ?"
        params.append(carrier.upper())
    if origin:
        query += " AND origin = ?"
        params.append(origin.upper())
    if dest:
        query += " AND dest = ?"
        params.append(dest.upper())
    if tail_num:
        query += " AND tail_num = ?"
        params.append(tail_num.upper())
    if risk_category:
        query += " AND risk_category = ?"
        params.append(risk_category.upper())
        
    query += " ORDER BY delay_prob DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/predict-flight")
def predict_new_flight(req: FlightInputRequest):
    """
    Interactive endpoint to input any custom flight schedule and get:
    - classification delay probability (DEP_DEL15)
    - regression delay duration in minutes (DEP_DELAY)
    - estimated actual arrival time
    - risk level
    - explainable factors / reasons for delay
    - full downstream aircraft rotation from the dataset
    """
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = FDPISPredictor(model_dir=MODELS_DIR)
        
    flight_dict = {
        "OP_UNIQUE_CARRIER": req.carrier.upper(),
        "ORIGIN": req.origin.upper(),
        "DEST": req.dest.upper(),
        "CRS_DEP_TIME": req.crs_dep_time,
        "CRS_ARR_TIME": req.crs_arr_time,
        "DAY_OF_MONTH": req.day_of_month,
        "DAY_OF_WEEK": req.day_of_week,
        "DISTANCE": req.distance,
        "TAIL_NUM": req.tail_num.upper(),
        "turnaround_buffer_min": req.turnaround_buffer_min
    }
    
    prediction_result = PREDICTOR.predict_flight(flight_dict)
    
    # Check if this aircraft has downstream scheduled legs on the same day in the database
    conn = get_db()
    cursor = conn.cursor()
    downstream_legs = cursor.execute("""
        SELECT * FROM flights 
        WHERE tail_num = ? AND day_of_month = ? AND crs_dep_time > ?
        ORDER BY crs_dep_time ASC
    """, [req.tail_num.upper(), req.day_of_month, req.crs_dep_time]).fetchall()
    conn.close()
    
    downstream_list = [dict(r) for r in downstream_legs]
    
    # Compute multi-leg cascade progression along rotation
    rotation_cascade = []
    incoming_delay = prediction_result["predicted_delay_minutes"]
    current_arr_time = req.crs_arr_time
    
    # Helper to convert HHMM to total minutes
    def hhmm_to_min(t):
        return (int(t) // 100) * 60 + (int(t) % 100)
        
    curr_arr_min = hhmm_to_min(current_arr_time)
    
    for leg in downstream_list:
        leg_dep_min = hhmm_to_min(leg["crs_dep_time"])
        ground_gap = leg_dep_min - curr_arr_min
        if ground_gap < -720:
            ground_gap += 1440
            
        absorbable_slack = max(0, ground_gap - 45) # 45m standard turnaround
        transmitted_delay = incoming_delay * 0.95
        residual_delay = max(0.0, transmitted_delay - absorbable_slack)
        
        # Calculate downstream estimated arrival
        leg_arr_min = hhmm_to_min(leg["crs_arr_time"]) + int(round(residual_delay))
        est_leg_arr_time = f"{(leg_arr_min // 60) % 24:02d}:{leg_arr_min % 60:02d}"
        
        rotation_cascade.append({
            "flight_id": leg["flight_id"],
            "route": f"{leg['origin']} -> {leg['dest']}",
            "sched_dep": f"{leg['crs_dep_time'] // 100:02d}:{leg['crs_dep_time'] % 100:02d}",
            "sched_arr": f"{leg['crs_arr_time'] // 100:02d}:{leg['crs_arr_time'] % 100:02d}",
            "est_arr_time": est_leg_arr_time,
            "ground_gap_min": ground_gap,
            "absorbable_slack_min": absorbable_slack,
            "incoming_delay_min": round(transmitted_delay, 1),
            "residual_delay_min": round(residual_delay, 1),
            "status": "DELAY_ABSORBED" if residual_delay <= 0 else ("HIGH_CASCADE_RISK" if residual_delay >= 30 else "MODERATE_CASCADE_RISK")
        })
        
        # Next flight inherits this residual delay
        incoming_delay = residual_delay
        curr_arr_min = hhmm_to_min(leg["crs_arr_time"])
        
    prediction_result["aircraft_tail"] = req.tail_num.upper()
    prediction_result["rotation_cascade"] = rotation_cascade
    return prediction_result

@app.get("/api/flights/{flight_id}")
def get_flight_detail(flight_id: str):
    """
    Get full flight profile, prediction breakdown, turnaround slack, and rotation schedule.
    """
    conn = get_db()
    cursor = conn.cursor()
    
    flight = cursor.execute("SELECT * FROM flights WHERE flight_id = ?", [flight_id]).fetchone()
    if not flight:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found.")
        
    flight_dict = dict(flight)
    
    rotation = []
    if flight_dict["tail_num"] and flight_dict["tail_num"] != "UNKNOWN":
        rot_rows = cursor.execute("""
            SELECT * FROM flights 
            WHERE tail_num = ? AND day_of_month = ?
            ORDER BY crs_dep_time ASC
        """, [flight_dict["tail_num"], flight_dict["day_of_month"]]).fetchall()
        rotation = [dict(r) for r in rot_rows]
        
    conn.close()
    return {
        "flight": flight_dict,
        "aircraft_rotation": rotation
    }

@app.get("/api/propagation/{flight_id}")
def get_flight_propagation_tree(flight_id: str, depth: int = 6):
    """
    Builds the dynamic network propagation tree for a root delayed flight using Breadth-First Search.
    """
    conn = get_db()
    cursor = conn.cursor()
    
    root_flight = cursor.execute("SELECT * FROM flights WHERE flight_id = ?", [flight_id]).fetchone()
    if not root_flight:
        conn.close()
        raise HTTPException(status_code=404, detail="Flight not found")
        
    root_dict = dict(root_flight)
    day = root_dict["day_of_month"]
    
    day_flights = cursor.execute("SELECT * FROM flights WHERE day_of_month = ?", [day]).fetchall()
    conn.close()
    
    df_day = pd.DataFrame([dict(r) for r in day_flights])
    df_day["CRS_DEP_TIME"] = df_day["crs_dep_time"]
    df_day["CRS_ARR_TIME"] = df_day["crs_arr_time"]
    df_day["OP_UNIQUE_CARRIER"] = df_day["carrier"]
    df_day["ORIGIN"] = df_day["origin"]
    df_day["DEST"] = df_day["dest"]
    df_day["TAIL_NUM"] = df_day["tail_num"]
    df_day["DISTANCE"] = df_day["distance"]
    
    engine = FDPISPropagationEngine(turnaround_buffer_threshold=45)
    engine.build_network_graph(df_day)
    
    init_delay = root_dict["predicted_delay"] if root_dict["predicted_delay"] > 0 else (45.0 if root_dict["delay_prob"] >= 0.40 else 25.0)
    tree = engine.propagate_cascade(flight_id, initial_delay_minutes=init_delay, max_depth=depth)
    return tree

@app.get("/api/model-info")
def get_model_info():
    """
    Returns verified model metrics, feature rankings, and evaluation statistics.
    """
    eval_path = os.path.join(RESULTS_DIR, "final_test_evaluation.json")
    reg_path = os.path.join(RESULTS_DIR, "final_test_regression.json")
    rank_path = os.path.join(RESULTS_DIR, "ranking_evaluation.json")
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    feat_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    
    eval_data = json.load(open(eval_path)) if os.path.exists(eval_path) else {}
    reg_data = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    rank_data = json.load(open(rank_path)) if os.path.exists(rank_path) else {}
    
    comp_table = pd.read_csv(comp_path).to_dict(orient="records") if os.path.exists(comp_path) else []
    feat_table = pd.read_csv(feat_path).head(15).to_dict(orient="records") if os.path.exists(feat_path) else []
    
    return {
        "classifier_test_metrics": eval_data,
        "regressor_test_metrics": reg_data,
        "ranking_performance": rank_data,
        "model_comparison_leaderboard": comp_table,
        "top_features": feat_table,
        "operational_profiles": {
            "competition_mode": {
                "threshold": 0.48,
                "accuracy": "77.44%",
                "precision": "68.60%",
                "recall": "11.79%",
                "description": "Calibrated for maximum classification accuracy"
            },
            "operations_mode": {
                "threshold": 0.18,
                "accuracy": "61.15%",
                "precision": "34.54%",
                "recall": "65.47%",
                "description": "Calibrated for maximum operational recall and F1 early warning"
            }
        }
    }

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "FDPIS Operations Engine API is running. Visit /docs for OpenAPI specifications."}
