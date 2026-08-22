# Flight Delay Propagation Intelligence System (FDPIS) — ML Engine

## 🏆 Model Competition Results

```
BEST MODEL:
Ensemble Blend (LightGBM + XGBoost + CatBoost)

BEST VALIDATION ACCURACY:
77.26%

FINAL TEST ACCURACY:
77.44%

FEATURE COUNT:
34

TOP FEATURES:
1. origin_daily_sched_flights (Origin airport daily congestion volume)
2. dest_daily_sched_flights (Destination airport daily volume)
3. turnaround_buffer_min (Aircraft inbound turnaround slack buffer in minutes)
4. carrier_origin_flights_today (Carrier operational scale at departure hub)
5. hist_delay_rate_ORIGIN_x_sched_dep_hour (Origin airport hourly historical delay rate)
6. hist_delay_rate_route (Historical origin-destination route delay rate)
7. carrier_origin_market_share (Carrier departure market share at origin)
8. hist_delay_rate_OP_UNIQUE_CARRIER_x_ORIGIN (Carrier-hub interaction delay baseline)
9. dow_sin (Day-of-week cyclical temporal demand)
10. hist_delay_rate_ORIGIN (Origin airport base delay rate)

WHY THIS MODEL WON:
The soft-voting ensemble combines the orthogonal strengths of XGBoost (regularized depth splits), LightGBM (leaf-wise continuous congestion partitioning), and CatBoost (smooth categorical feature representations). Calibrating the optimal decision threshold (tau = 0.48) on the validation set maximized true classifications while minimizing false alarms, achieving 77.44% accuracy on 78,904 out-of-time test flights.

LEAKAGE CHECK:
Confirmed Clean. Excluded all 21 post-departure realization fields (DEP_TIME, ARR_TIME, TAXI_OUT/IN, WHEELS_OFF/ON, actual delays, cancellations, diversion flags, and delay cause codes). Historical delay statistics strictly computed from the training split using Bayesian prior smoothing.
```

---

## 📊 Comprehensive Model Comparison Table (Validation Set: Days 22–26)

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

## 🧪 Locked Test Set Performance (Days 27–31, $N = 78,904$)

- **Final Test Accuracy:** **77.44%**
- **Precision:** 68.60%
- **Recall:** 11.79%
- **F1-Score:** 20.12%
- **ROC-AUC:** 0.6440
- **PR-AUC:** 0.4244
- **Confusion Matrix:**
  $$\begin{bmatrix} 58,864 & 1,026 \\ 16,773 & 2,241 \end{bmatrix}$$

---

## 📂 Project Structure

```
FDPIS Model Testing/
├── models/
│   ├── lightgbm.pkl              # Trained LightGBM model artifact
│   ├── xgboost.pkl               # Trained XGBoost model artifact
│   ├── catboost.pkl              # Trained CatBoost model artifact
│   ├── scaler.pkl                # StandardScaler for linear baselines
│   ├── features.json             # 34 engineered feature names
│   ├── historical_stats.json     # Bayesian smoothed delay rate mappings
│   └── model_metadata.json       # Optimal thresholds & evaluation metrics
├── results/
│   ├── model_comparison.csv      # Validation comparison results
│   ├── feature_importance.csv    # Tree feature importances
│   └── final_test_evaluation.json # Final locked test evaluation metrics
├── src/
│   ├── feature_pipeline.py       # Zero-leakage ETL & feature engineering
│   ├── train_models.py           # Multi-model training and evaluation suite
│   ├── evaluate_test.py          # Final locked test set evaluation
│   ├── predict.py                # Standalone inference class
│   ├── error_analysis.py         # Breakdown of errors by hub, carrier, hour
│   └── audit_dataset.py          # Dataset inspection and statistics audit
├── .gitignore                    # Prevents raw large datasets from polluting git
└── README.md                     # Model documentation and reproduction guide
```

---

## 🚀 How to Run Inference

```python
from src.predict import FDPISPredictor
import pandas as pd

predictor = FDPISPredictor(model_dir="models")

# Provide pre-departure flight schedule parameters
sample_flight = pd.DataFrame([{
    "DAY_OF_MONTH": 15,
    "DAY_OF_WEEK": 4,
    "CRS_DEP_TIME": 830,        # 08:30 AM scheduled departure
    "CRS_ARR_TIME": 1100,       # 11:00 AM scheduled arrival
    "CRS_ELAPSED_TIME": 150,
    "DISTANCE": 950,
    "OP_UNIQUE_CARRIER": "DL",
    "ORIGIN": "ATL",
    "DEST": "LGA",
    "TAIL_NUM": "N901DA"
}])

pred_label = predictor.predict(sample_flight)
pred_prob = predictor.predict_proba(sample_flight)

print(f"Delay Prediction: {'Delayed (>=15 min)' if pred_label[0] == 1 else 'On-Time'}")
print(f"Delay Probability: {pred_prob[0]*100:.1f}%")
```

---

## 🔄 Push to GitHub

```bash
cd "C:\Users\ashis\Desktop\Random Projs\FDPIS Model Testing"
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```
