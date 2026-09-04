# Flight Delay Propagation Intelligence System (FDPIS) — End-to-End Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Framework](https://img.shields.io/badge/Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-blue.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-77.26%25-green.svg)]()
[![Test Accuracy](https://img.shields.io/badge/Final%20Test%20Accuracy-77.44%25-green.svg)]()
[![Delay Regressor MAE](https://img.shields.io/badge/Delay%20Regressor%20MAE-20.07%20min-blue.svg)]()

The Flight Delay Propagation Intelligence System (FDPIS) is a production-grade, network-aware aviation analytics and machine learning platform. It combines gradient boosting classification, Huber-loss delay magnitude regression, and a graph propagation engine (Breadth-First Search over aircraft rotations) to forecast flight delays and downstream cascade risks 3 to 6 hours before departure.

---

## 1. System Architecture (5 Layers)

```
[ Layer 1: Data Ingestion & Chronological Partitions ]
  BTS Flight Performance Data (517,222 Valid Records, Jan 2026)
  Train (Days 1-21) | Val (Days 22-26) | Locked Test (Days 27-31)
                    │
                    ▼
[ Layer 2: Pre-Departure Feature Engineering ]
  34 Verified Leakage-Free Features:
  - Temporal & Cyclical (Hour/Day Sin/Cos, Time-of-Day Buckets)
  - Hub & Route Congestion (Hourly Departures, Carrier Market Share)
  - Inbound Aircraft Turnaround Buffer (Sched Dep - Inbound Sched Arr)
  - Bayesian Smoothed Delay Rates (Computed strictly on Train set)
                    │
                    ▼
[ Layer 3: Dual ML Prediction Engine ]
  ┌─────────────────────────────────┬─────────────────────────────────┐
  │      Classifier (DEP_DEL15)     │    Regressor (Delay Magnitude)  │
  │ Ensemble (LGBM + XGB + CatBoost)│  LightGBM Huber-Loss Regressor  │
  │ • Competition: 77.44% Accuracy  │  • Test MAE: 20.07 min          │
  │ • Operations: 65.47% Recall     │  • Test MedAE: 0.95 min         │
  └─────────────────────────────────┴─────────────────────────────────┘
                    │
                    ▼
[ Layer 4: Network Propagation Engine (BFS Graph) ]
  Directed Acyclic Rotation Chains:
  Transmitted Delay = (Incoming Delay * 0.95) - Absorbable Slack
                    │
                    ▼
[ Layer 5: Operational Decision & Recommendation Layer ]
  - FastAPI Asynchronous REST Engine
  - Rule-Based Advisory Alerts (DGCA/FDTL Crew Duty Limits, Turnaround Compression)
  - Interactive Operations Dashboard (Vis.js Cascade Tree Visualizer)
```

---

## 2. ML Performance & Benchmark Results

### A. Classification Suite (Target: `DEP_DEL15`)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Blend (LightGBM + XGB + CatBoost)** | **77.26%** | **67.98%** | **13.51%** | **0.2254** | **0.6794** | **0.48** |
| **XGBoost Classifier** | 77.24% | 69.48% | 12.56% | 0.2127 | 0.6801 | 0.51 |
| **LightGBM Classifier** | 77.19% | 66.61% | 13.76% | 0.2281 | 0.6782 | 0.48 |
| **CatBoost Classifier** | 77.17% | 67.98% | 12.83% | 0.2159 | 0.6761 | 0.48 |
| **HistGradientBoosting** | 77.12% | 75.81% | 09.62% | 0.1708 | 0.6756 | 0.56 |
| **Logistic Regression (Scaled)** | 76.34% | 64.11% | 07.69% | 0.1373 | 0.6609 | 0.56 |
| **Majority Class Baseline** | 75.51% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.50 |

#### Dual Operational Calibration Profiles
- **Profile 1 (Competition Mode, Threshold = 0.48):** Optimized for raw classification accuracy.
  - Final Test Accuracy: **77.44%** | Precision: **68.60%** | Recall: **11.79%**
- **Profile 2 (Operations Mode, Threshold = 0.18):** Optimized for operational early warning & balanced F1.
  - Validation Accuracy: **61.15%** | Precision: **34.54%** | Recall: **65.47%** | F1: **0.4522**

#### Operational Inspection Queue Ranking (Precision@K%)
- **Precision@0.5% (top 355 flights):** **98.03%** (Lift: 4.00x over random)
- **Precision@1.0% (top 711 flights):** **96.34%** (Lift: 3.93x over random)
- **Precision@2.0% (top 1,422 flights):** **87.83%** (Lift: 3.59x over random)
- **Precision@5.0% (top 3,557 flights):** **67.50%** (Lift: 2.76x over random)

---

### B. Delay Duration Regression Suite (Target: `DEP_DELAY` in Minutes)

| Model Architecture | Validation MAE | Validation RMSE | Validation MedAE |
| :--- | :---: | :---: | :---: |
| **LightGBM Regressor (Huber Loss)** | **22.32 min** | **79.03 min** | **1.07 min** |
| **CatBoost Regressor (MAE Loss)** | 22.53 min | 79.72 min | 0.40 min |
| **Ridge Regression Baseline** | 23.22 min | 79.95 min | 2.39 min |
| **Median Delay Baseline** | 23.40 min | 81.75 min | 0.00 min |
| **Mean Delay Baseline** | 28.63 min | 78.85 min | 14.35 min |

**Final Locked Test Set Regression (LightGBM):**
- **Test MAE:** **20.07 minutes**
- **Test RMSE:** **68.77 minutes**
- **Test Median Absolute Error (MedAE):** **0.95 minutes**

---

## 3. Directory Structure

```
.
├── data/
│   └── fdpis_flights.db          # Indexed SQLite operational database (149,982 flights)
├── models/
│   ├── lightgbm.pkl              # Primary LightGBM classifier
│   ├── xgboost.pkl               # Primary XGBoost classifier
│   ├── catboost.pkl              # Primary CatBoost classifier
│   ├── delay_regressor.pkl       # Final Huber-loss delay duration regressor
│   ├── features.json             # 34 engineered feature names
│   ├── historical_stats.json     # Bayesian smoothed delay rate lookup tables
│   └── model_metadata.json       # Thresholds and test evaluation metrics
├── results/
│   ├── final_test_evaluation.json # Test classifier metrics
│   ├── final_test_regression.json # Test regressor metrics
│   ├── ranking_evaluation.json   # Precision@K% queue rankings
│   ├── threshold_tradeoff.csv    # Accuracy vs Recall vs F1 threshold sweep
│   ├── feature_importance.csv    # Feature contribution rankings
│   └── plots/
│       ├── feature_importance.png # Feature importance chart
│       └── threshold_tradeoff.png # Trade-off curve visualization
├── src/
│   ├── app.py                    # FastAPI application & REST endpoints
│   ├── propagation_engine.py     # Graph construction & BFS cascade traversal
│   ├── database.py               # SQLite schema & indexing script
│   ├── populate_db.py            # Batch inference & database populator
│   ├── feature_pipeline.py       # Zero-leakage ETL & feature extraction
│   ├── train_models.py           # Classification model training suite
│   ├── train_regressor.py        # Regression training & evaluation suite
│   ├── tune_classifier.py        # Threshold sweep & ranking analysis
│   ├── evaluate_test.py          # Final locked test evaluation
│   ├── predict.py                # Standalone inference class
│   └── cli_predict.py            # Command-line interface for predictions
├── static/
│   ├── index.html                # Interactive Operations Intelligence Dashboard
│   └── plots/                    # Web-accessible diagnostic plots
├── tests/
│   └── test_system.py            # Automated test suite (API, ML, Graph)
├── requirements.txt              # Production dependency list
├── .gitignore
└── README.md
```

---

## 4. How to Run the Platform

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated System Tests
```bash
python -m unittest discover tests
```
*All 5 unit tests pass, confirming API routing, model artifact integrity, and BFS graph cascade logic.*

### Step 3: Launch the Production Web Dashboard & API
Start the FastAPI server using Uvicorn:
```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
- **Operations Dashboard:** `http://127.0.0.1:8000/`
- **Interactive OpenAPI Documentation:** `http://127.0.0.1:8000/docs`

---

## 5. Using the Interactive Dashboard

1. **Operations Overview:** Displays top monitored KPIs, critical origin hubs facing congestion, and highest-risk flights requiring immediate review.
2. **Flight Intelligence:** Filter 150k operational flights by carrier, origin, destination, aircraft tail number, and risk category.
3. **Cascade Propagation Tree:** Click **"Trace Cascade"** on any flight to inspect the Breadth-First Search (BFS) graph. The visualization illustrates how delays propagate across aircraft rotations, where buffer slack absorbs delays, and triggers rule-based DGCA/FDTL crew duty alerts.
4. **Model Diagnostics:** View diagnostic plots for feature importance and threshold trade-offs directly in the interface.

---

## 6. License
Released under the MIT License.
