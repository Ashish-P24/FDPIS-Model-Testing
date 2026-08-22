# Flight Delay Propagation Intelligence System (FDPIS) — ML Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Framework](https://img.shields.io/badge/Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-blue.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-77.26%25-green.svg)]()
[![Test Accuracy](https://img.shields.io/badge/Final%20Test%20Accuracy-77.44%25-green.svg)]()

A machine learning engine developed for the Flight Delay Propagation Intelligence System (FDPIS) project to predict primary flight departure delays (`DEP_DEL15`) and network cascade risks using US Bureau of Transportation Statistics (BTS) on-time performance records.

---

## 1. Project Overview & Objective

The primary objective of this project is to construct a verified, leakage-free machine learning model achieving the highest legitimate classification accuracy on commercial flight departure delays.

### Core Problem
Flight delays cost the global aviation industry tens of billions of dollars annually. Early prediction of primary flight delays (3 to 6 hours before departure) allows Operations Control Centres (OCC) to proactively manage aircraft rotations, reposition standby crew, and mitigate downstream network delay propagation.

### Target Definition
- **Target Variable:** `DEP_DEL15`
- **Class 1 (Delayed):** Departure delay $\ge 15$ minutes
- **Class 0 (On-time):** Departure delay $< 15$ minutes
- **Evaluation Benchmark:** Classification Accuracy on an untouched, out-of-time chronological test set.

---

## 2. Model Performance & Leaderboard

All candidate models were trained on the training partition (Days 1–21) and systematically evaluated on the validation partition (Days 22–26):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Blend (LightGBM + XGB + CatBoost)** | **77.26%** | **67.98%** | **13.51%** | **0.2254** | **0.6794** | **0.48** |
| **XGBoost Classifier** | 77.24% | 69.48% | 12.56% | 0.2127 | 0.6801 | 0.51 |
| **LightGBM Classifier** | 77.19% | 66.61% | 13.76% | 0.2281 | 0.6782 | 0.48 |
| **CatBoost Classifier** | 77.17% | 67.98% | 12.83% | 0.2159 | 0.6761 | 0.48 |
| **HistGradientBoosting** | 77.12% | 75.81% | 09.62% | 0.1708 | 0.6756 | 0.56 |
| **Logistic Regression (StandardScaled)** | 76.34% | 64.11% | 07.69% | 0.1373 | 0.6609 | 0.56 |
| **Majority Class Baseline** | 75.51% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.50 |

---

## 3. Locked Test Set Results (Days 27–31, $N = 78,904$)

The winning ensemble was evaluated on the locked, untouched test partition:

- **Final Test Accuracy:** **77.44%** *(+1.54% over majority-class baseline of 75.90%)*
- **Precision:** 68.60% *(low false-alarm rate)*
- **Recall:** 11.79%
- **F1-Score:** 20.12%
- **ROC-AUC:** 0.6440
- **PR-AUC:** 0.4244
- **Calibrated Decision Threshold:** $\tau^* = 0.48$

### Confusion Matrix
```
                    Predicted On-Time    Predicted Delayed (>=15 min)
Actual On-Time:          58,864                      1,026
Actual Delayed:          16,773                      2,241
```

---

## 4. Zero Data Leakage Protocol

To ensure realistic, deployable performance, this pipeline strictly excludes post-departure realization fields:

1. **Excluded Post-Event Fields (21 columns):**
   - Actual timestamps: `DEP_TIME`, `ARR_TIME`, `WHEELS_OFF`, `WHEELS_ON`
   - Realized taxi & flight times: `TAXI_OUT`, `TAXI_IN`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME`
   - Realized delay metrics: `DEP_DELAY`, `ARR_DELAY`, `ARR_DEL15`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`
   - Realized operational states: `CANCELLED`, `CANCELLATION_CODE`, `DIVERTED`

2. **Temporal Splitting:**
   - **Train (Days 1–21):** 367,169 flights (71.0%)
   - **Validation (Days 22–26):** 71,149 flights (13.8%)
   - **Locked Test (Days 27–31):** 78,904 flights (15.3%)

3. **Indirect Leakage Prevention:**
   - Historical delay rates for routes, carriers, and airports were computed **strictly on the training split** and mapped to validation/test using Bayesian prior smoothing ($m=10$).
   - Aircraft turnaround slack is computed strictly from the preceding scheduled leg with scheduled arrival prior to current departure.

---

## 5. Feature Engineering Summary

The model uses 34 domain-engineered features categorized into four functional groups:

1. **Temporal & Schedule Dynamics:**
   - Scheduled departure and arrival hour/minute/total minutes
   - Cyclical hour encoding: $\sin(2\pi \cdot \text{hour} / 24)$, $\cos(2\pi \cdot \text{hour} / 24)$
   - Day of week cyclical encoding: $\sin(2\pi \cdot \text{DOW} / 7)$, $\cos(2\pi \cdot \text{DOW} / 7)$
   - Weekend indicator and operational time-of-day buckets
   - Scheduled block speed proxy ($\text{distance} / \text{elapsed\_time}$)

2. **Airport & Hub Congestion Proxies:**
   - Scheduled departures at Origin airport within hourly window
   - Daily scheduled departure volume at Origin airport
   - Daily scheduled arrival volume at Destination airport
   - Carrier departure count and market share percentage at Origin airport

3. **Aircraft Turnaround & Propagation (FDPIS Core):**
   - Scheduled turnaround buffer in minutes ($\text{CRS\_DEP\_TIME}_{\text{curr}} - \text{CRS\_ARR\_TIME}_{\text{prev}}$)
   - Tight turnaround risk indicator (slack $< 45$ minutes)
   - Negative buffer indicator (inbound scheduled arrival exceeds scheduled departure)
   - Daily flight sequence counter for the aircraft tail number

4. **Bayesian Historical Encodings:**
   - Carrier historical delay rate
   - Origin airport historical delay rate
   - Destination airport historical delay rate
   - Route historical delay rate
   - Origin $\times$ Departure Hour interaction delay rate
   - Carrier $\times$ Origin Hub interaction delay rate

---

## 6. Directory Structure

```
.
├── models/
│   ├── lightgbm.pkl              # Trained LightGBM model binary
│   ├── xgboost.pkl               # Trained XGBoost model binary
│   ├── catboost.pkl              # Trained CatBoost model binary
│   ├── scaler.pkl                # Standard scaler for linear baseline
│   ├── features.json             # List of 34 engineered feature names
│   ├── historical_stats.json     # Bayesian smoothed delay rate lookup tables
│   └── model_metadata.json       # Thresholds and test evaluation metrics
├── results/
│   ├── model_comparison.csv      # Validation comparison table
│   ├── feature_importance.csv    # Feature contribution rankings
│   └── final_test_evaluation.json # Final locked test evaluation report
├── src/
│   ├── feature_pipeline.py       # Data cleaning and feature engineering
│   ├── train_models.py           # Model training and validation suite
│   ├── evaluate_test.py          # Final locked test set evaluation
│   ├── predict.py                # Standalone inference class (FDPISPredictor)
│   ├── cli_predict.py            # Command-line interface for predictions
│   ├── error_analysis.py         # Breakdown of errors across hubs and carriers
│   ├── hyperopt_tune.py          # Optuna Bayesian hyperparameter optimization
│   └── audit_dataset.py          # Dataset inspection and statistics audit
├── .gitignore                    # Git ignore file
└── README.md                     # Documentation
```

---

## 7. How to Use the Program

### Step 1: Environment Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Ashish-P24/FDPIS-Model-Testing.git
cd FDPIS-Model-Testing

# Install required Python packages
pip install pandas scikit-learn xgboost lightgbm catboost optuna pyarrow
```

---

### Step 2: Dataset Preparation
Ensure the raw BTS on-time reporting CSV file (`T_ONTIME_REPORTING.csv`) is available. By default, `src/feature_pipeline.py` references the local dataset path:
```python
DATA_PATH = r"C:\Users\ashis\Downloads\T_ONTIME_REPORTING_20260820_081428\T_ONTIME_REPORTING.csv"
```
*(Update `DATA_PATH` in `src/feature_pipeline.py` if your file is located elsewhere).*

---

### Step 3: Run Feature Engineering
Execute the feature pipeline to clean the raw data, generate the 34 predictive features, compute historical target encodings, and generate chronological partitions:

```bash
python src/feature_pipeline.py
```
*Output: Generates `results/train.parquet`, `results/val.parquet`, `results/test.parquet`, and `models/historical_stats.json`.*

---

### Step 4: Train All Models & Ensembles
Train the baseline models, gradient boosting classifiers, and the Soft-Voting Ensemble:

```bash
python src/train_models.py
```
*Output: Trains all models, saves binaries to `models/`, tunes the optimal probability threshold, and outputs the model validation comparison table.*

---

### Step 5: Evaluate on Locked Test Set
Evaluate the winning ensemble on the locked out-of-time test partition:

```bash
python src/evaluate_test.py
```
*Output: Reports the final test accuracy (77.44%), precision, recall, F1, ROC-AUC, and confusion matrix.*

---

### Step 6: Make Single-Flight Predictions via CLI
Use the command-line tool `cli_predict.py` to predict delays for any scheduled flight:

```bash
python src/cli_predict.py --carrier DL --origin ATL --dest LGA --dep_time 830 --arr_time 1100 --distance 762 --elapsed_time 150 --day_of_month 15 --day_of_week 4 --tail_num N901DA
```

**Sample Output:**
```
=======================================================
      FDPIS FLIGHT DELAY PREDICTION RESULT
=======================================================
 Flight:       DL | ATL -> LGA
 Schedule:     Dep 0830 | Arr 1100 | Day 15
 Aircraft:     N901DA
-------------------------------------------------------
 Delay Probability:        18.42%
 Calibrated Threshold:     0.48
 Prediction Status:        ON-TIME (< 15 minutes)
 Risk Assessment:          LOW CASCADE RISK
=======================================================
```

---

### Step 7: Integrate Predictions in Python Code
You can import the `FDPISPredictor` class directly into any Python script or pipeline:

```python
import pandas as pd
from src.predict import FDPISPredictor

# Initialize the predictor
predictor = FDPISPredictor(model_dir="models")

# Define flight parameters
flight = pd.DataFrame([{
    "DAY_OF_MONTH": 20,
    "DAY_OF_WEEK": 5,
    "CRS_DEP_TIME": 1730,       # 05:30 PM
    "CRS_ARR_TIME": 2015,       # 08:15 PM
    "CRS_ELAPSED_TIME": 165,
    "DISTANCE": 1020,
    "OP_UNIQUE_CARRIER": "AA",
    "ORIGIN": "ORD",
    "DEST": "MIA",
    "TAIL_NUM": "N802AA"
}])

# Get binary classification (1 = Delayed >= 15 min, 0 = On-Time)
prediction = predictor.predict(flight)

# Get probability score (0.0 to 1.0)
probability = predictor.predict_proba(flight)

print("Predicted Class:", "Delayed" if prediction[0] == 1 else "On-Time")
print(f"Delay Probability: {probability[0]*100:.2f}%")
```

---

## 8. License
This project is released under the MIT License.
