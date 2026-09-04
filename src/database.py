import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DB_PATH = os.path.join(BASE_DIR, "data", "fdpis_flights.db")

def initialize_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print(f"Initializing SQLite operational database at {DB_PATH}...", flush=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create flights table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flights (
        flight_id TEXT PRIMARY KEY,
        carrier TEXT,
        tail_num TEXT,
        origin TEXT,
        dest TEXT,
        day_of_month INTEGER,
        day_of_week INTEGER,
        crs_dep_time INTEGER,
        crs_arr_time INTEGER,
        distance REAL,
        sched_elapsed_time REAL,
        delay_prob REAL,
        predicted_delay REAL,
        risk_category TEXT,
        is_delayed_actual INTEGER,
        dep_delay_actual REAL
    );
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_carrier ON flights(carrier);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_origin ON flights(origin);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dest ON flights(dest);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tail ON flights(tail_num);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_day ON flights(day_of_month);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk ON flights(risk_category);")
    
    conn.commit()
    conn.close()
    print("Database tables and indexes created successfully.")

if __name__ == "__main__":
    initialize_database()
