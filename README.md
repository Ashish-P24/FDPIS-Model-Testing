# ✈️ Flight Delay Propagation Intelligence System (FDPIS) — ML Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Framework](https://img.shields.io/badge/Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-success.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-77.26%25-brightgreen.svg)]()
[![Test Accuracy](https://img.shields.io/badge/Final%20Test%20Accuracy-77.44%25-brightgreen.svg)]()

An advanced, production-grade Machine Learning engine designed to predict primary flight departure delays (`DEP_DEL15`) and network delay cascades using US Bureau of Transportation Statistics (BTS) on-time performance data.

---

## 🏆 Model Competition Leaderboard

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Ensemble Blend (LightGBM + XGB + CatBoost)** | **77.26%** | **67.98%** | **13.51%** | **0.2254** | **0.6794** | **0.48** |
| 🥈 **XGBoost Classifier** | 77.24% | 69.48% | 12.56% | 0.2127 | 0.6801 | 0.51 |
| 🥉 **LightGBM Classifier** | 77.19% | 66.61% | 13.76% | 0.2281 | 0.6782 | 0.48 |
| 4. **CatBoost Classifier** | 77.17% | 67.98% | 12.83% | 0.2159 | 0.6761 | 0.48 |
| 5. **HistGradientBoosting** | 77.12% | 75.81% | 09.62% | 0.1708 | 0.6756 | 0.56 |
| 6. **Logistic Regression (StandardScaled)** | 76.34% | 64.11% | 07.69% | 0.1373 | 0.6609 | 0.56 |
| 7. **Majority Class Baseline** | 75.51% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.50 |

---

## 🎯 Locked Test Set Evaluation Results (Days 27–31, $N = 78,904$)

The winning ensemble was evaluated on the **locked out-of-time test partition** representing the final 5 operating days of January 2026:

- **FINAL TEST ACCURACY:** **`77.44%`** *(beats majority-class baseline of 75.90% by +1.54% real accuracy points)*
- **Precision:** `68.60%` *(low false-alarm rate essential for airline Operations Control Centres)*
- **Recall:** `11.79%`
- **F1-Score:** `20.12%`
- **ROC-AUC:** `0.6440`
- **PR-AUC:** `0.4244`
- **Optimal Probability Threshold:** $\tau^* = 0.48$

### 📉 Confusion Matrix
```
                    Predicted On-Time    Predicted Delayed (>=15 min)
Actual On-Time:          58,864                      1,026
Actual Delayed:          16,773                      2,241
```

---

## 🏗️ End-to-End System Architecture

```
                                  BTS Flight Dataset (544,003 Records)
                                                  │
                                                  ▼
                                 ┌──────────────────────────────────┐
                                 │   Data Cleaning & Validation     │
                                 │  (Filter Cancelled & Diverted)   │
                                 └──────────────────────────────────┘
                                                  │
                                                  ▼
                                 ┌──────────────────────────────────┐
                                 │    Zero-Leakage Feature Engine   │
                                 │  (34 Aviation Engineered Feats)  │
                                 └──────────────────────────────────┘
                                                  │
                                                  ▼
                     ┌────────────────────────────┼────────────────────────────┐
                     │ (Days 1–21)                │ (Days 22–26)               │ (Days 27–31)
                     ▼                            ▼                            ▼
             Train Set (71.0%)             Val Set (13.8%)             Locked Test Set (15.3%)
             [367,169 Flights]             [71,149 Flights]                [78,904 Flights]
                     │                            │                                │
                     ▼                            ▼                                │
            ┌──────────────────┐         ┌──────────────────┐                      │
            │ Train Candidates │ ──────▶ │ Threshold Tuning │                      │
            │ (LGBM, XGB, CB)  │         │ & Model Blending │                      │
            └──────────────────┘         └──────────────────┘                      │
                                                  │                                │
                                                  ▼                                ▼
                                       ┌──────────────────────────────────────────────────┐
                                       │       Final Evaluation & Saved Predictor         │
                                       │     Accuracy: 77.44% | Precision: 68.60%         │
                                       └──────────────────────────────────────────────────┘
```

---

## 🛡️ Zero Data Leakage Guarantee

In flight operations, predictions must occur **3 to 6 hours before departure**. This model strictly eliminates both direct and indirect target leakage:

1. **Direct Post-Departure Realization Columns Excluded (21 features):**
   - Actual timestamps: `DEP_TIME`, `ARR_TIME`, `WHEELS_OFF`, `WHEELS_ON`
   - Realized taxi & flight times: `TAXI_OUT`, `TAXI_IN`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME`
   - Realized delays & cause codes: `DEP_DELAY`, `ARR_DELAY`, `ARR_DEL15`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`
   - Flight status flags: `CANCELLED`, `CANCELLATION_CODE`, `DIVERTED`

2. **Indirect / Temporal Leakage Prevention:**
   - **Historical Delay Rates:** Route, carrier, and airport delay rates were computed **strictly on the training split** (Days 1–21) and mapped using Bayesian prior smoothing ($m=10$).
   - **Aircraft Turnaround Slack:** The turnaround buffer ($\text{CRS\_DEP\_TIME}_{\text{curr}} - \text{CRS\_ARR\_TIME}_{\text{prev}}$) strictly references the previous scheduled flight of the same physical aircraft (`TAIL_NUM`) where $\text{CRS\_ARR\_TIME}_{\text{prev}} < \text{CRS\_DEP\_TIME}_{\text{curr}}$.

---

## 🔍 Feature Importance Ranking

The top 10 most predictive features identified by gradient boosting feature attribution:

| Rank | Feature Name | Category | Description |
| :---: | :--- | :--- | :--- |
| 1 | `origin_daily_sched_flights` | Airport Congestion | Total scheduled departures at origin airport today |
| 2 | `dest_daily_sched_flights` | Airport Congestion | Total scheduled arrivals at destination airport today |
| 3 | `turnaround_buffer_min` | FDPIS Propagation | Scheduled slack time between inbound arrival and departure |
| 4 | `carrier_origin_flights_today` | Carrier Operations | Volume of carrier flights operated at departure airport |
| 5 | `hist_delay_rate_ORIGIN_x_sched_dep_hour` | Interaction | Historical delay rate for origin airport at specific departure hour |
| 6 | `hist_delay_rate_route` | Route Performance | Historical baseline delay frequency for the origin-destination pair |
| 7 | `carrier_origin_market_share` | Carrier Operations | Percentage of origin airport traffic controlled by the carrier |
| 8 | `hist_delay_rate_OP_UNIQUE_CARRIER_x_ORIGIN` | Interaction | Historical performance of carrier at specific departure hub |
| 9 | `dow_sin` | Temporal | Cyclical sine encoding of the day of the week |
| 10 | `hist_delay_rate_ORIGIN` | Airport Baseline | Overall historical departure delay rate of origin airport |

---

## 📁 Repository Structure

```
.
├── models/
│   ├── lightgbm.pkl              # Trained LightGBM model binary
│   ├── xgboost.pkl               # Trained XGBoost model binary
│   ├── catboost.pkl              # Trained CatBoost model binary
│   ├── scaler.pkl                # Standard scaler for linear models
│   ├── features.json             # 34 engineered feature specifications
│   ├── historical_stats.json     # Bayesian smoothed delay rate lookup tables
│   └── model_metadata.json       # Optimal thresholds & evaluation metrics
├── results/
│   ├── model_comparison.csv      # Validation comparison across all algorithms
│   ├── feature_importance.csv    # Feature contribution rankings
│   └── final_test_evaluation.json # Final locked test evaluation report
├── src/
│   ├── feature_pipeline.py       # Zero-leakage ETL, cleaning & feature engineering
│   ├── train_models.py           # Multi-model training and validation suite
│   ├── evaluate_test.py          # Final locked test set evaluation
│   ├── predict.py                # Standalone inference class (FDPISPredictor)
│   ├── error_analysis.py         # Breakdown of errors by carrier, hub, and hour
│   ├── hyperopt_tune.py          # Optuna Bayesian hyperparameter optimization
│   └── audit_dataset.py          # Dataset inspection and statistics audit
├── .gitignore                    # Prevents raw large datasets from polluting git
└── README.md                     # Comprehensive system documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites & Installation
```bash
git clone https://github.com/Ashish-P24/FDPIS-Model-Testing.git
cd FDPIS-Model-Testing
pip install pandas scikit-learn xgboost lightgbm catboost optuna pyarrow
```

### 2. Run the Full ML Pipeline
```bash
# Step 1: Clean data and engineer 34 predictive features
python src/feature_pipeline.py

# Step 2: Train baseline models, GBDTs, and Soft-Voting Ensemble
python src/train_models.py

# Step 3: Evaluate winning model on the locked test set
python src/evaluate_test.py
```

---

## 💡 Running Live Inference in Python

```python
import pandas as pd
from src.predict import FDPISPredictor

# Initialize the predictor (loads models, feature mappings & optimal threshold)
predictor = FDPISPredictor(model_dir="models")

# Provide pre-departure flight schedule parameters
sample_flight = pd.DataFrame([{
    "DAY_OF_MONTH": 15,
    "DAY_OF_WEEK": 4,
    "CRS_DEP_TIME": 830,        # 08:30 AM scheduled departure
    "CRS_ARR_TIME": 1100,       # 11:00 AM scheduled arrival
    "CRS_ELAPSED_TIME": 150,    # 150 minutes block time
    "DISTANCE": 950,            # 950 miles
    "OP_UNIQUE_CARRIER": "DL",  # Delta Air Lines
    "ORIGIN": "ATL",            # Atlanta Hartsfield-Jackson
    "DEST": "LGA",              # New York LaGuardia
    "TAIL_NUM": "N901DA"        # Aircraft tail number
}])

# Generate prediction and continuous delay probability
pred_label = predictor.predict(sample_flight)
pred_prob = predictor.predict_proba(sample_flight)

print(f"Prediction: {'⚠️ DELAYED (>=15 min)' if pred_label[0] == 1 else '✅ ON-TIME'}")
print(f"Delay Probability: {pred_prob[0] * 100:.2f}%")
```

---

## 🔮 Roadmap to Push Accuracy Beyond 80–85%+

1. **Weather Forecast Ingestion (NOAA METAR / TAF):** Weather accounts for $\sim 35\%$ of non-propagated delays. Integrating surface visibility, crosswinds, and convective storm flags will provide significant predictive lift.
2. **Real-Time ADS-B Aircraft Tracking:** Ingesting live airborne position 3 hours prior to scheduled departure detects late inbound airframes with $>85\%$ accuracy.
3. **Multi-Month Seasonal Data:** Training across 12 months captures summer convective patterns and major holiday surge traffic.
4. **Graph Propagation Engine (FDPIS Layer 4):** Integrating BFS dependency tree cascading scores directly into the feature matrix.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
