import argparse
import sys
import os
import pandas as pd
from predict import FDPISPredictor

def main():
    parser = argparse.ArgumentParser(description="FDPIS Flight Delay Predictor CLI")
    parser.add_argument("--carrier", type=str, required=True, help="Operating Carrier Code (e.g. DL, AA, WN, UA)")
    parser.add_argument("--origin", type=str, required=True, help="Origin Airport IATA Code (e.g. ATL, ORD, DFW)")
    parser.add_argument("--dest", type=str, required=True, help="Destination Airport IATA Code (e.g. LGA, LAX, MIA)")
    parser.add_argument("--dep_time", type=int, required=True, help="Scheduled Departure Time in HHMM format (e.g. 830 for 08:30 AM, 1645 for 04:45 PM)")
    parser.add_argument("--arr_time", type=int, required=True, help="Scheduled Arrival Time in HHMM format (e.g. 1100 for 11:00 AM)")
    parser.add_argument("--day_of_month", type=int, default=15, help="Day of the month (1-31)")
    parser.add_argument("--day_of_week", type=int, default=4, help="Day of the week (1=Mon, 7=Sun)")
    parser.add_argument("--distance", type=float, default=800.0, help="Flight distance in miles")
    parser.add_argument("--elapsed_time", type=float, default=120.0, help="Scheduled block time in minutes")
    parser.add_argument("--tail_num", type=str, default="UNKNOWN", help="Aircraft Tail Number (e.g. N901DA)")
    parser.add_argument("--model_dir", type=str, default="models", help="Path to models directory")

    args = parser.parse_args()

    model_dir_path = args.model_dir
    if not os.path.isabs(model_dir_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir_path = os.path.join(base_dir, model_dir_path)

    predictor = FDPISPredictor(model_dir=model_dir_path)

    flight_df = pd.DataFrame([{
        "DAY_OF_MONTH": args.day_of_month,
        "DAY_OF_WEEK": args.day_of_week,
        "CRS_DEP_TIME": args.dep_time,
        "CRS_ARR_TIME": args.arr_time,
        "CRS_ELAPSED_TIME": args.elapsed_time,
        "DISTANCE": args.distance,
        "OP_UNIQUE_CARRIER": args.carrier.upper(),
        "ORIGIN": args.origin.upper(),
        "DEST": args.dest.upper(),
        "TAIL_NUM": args.tail_num.upper()
    }])

    prob = predictor.predict_proba(flight_df)[0]
    pred = predictor.predict(flight_df)[0]

    print("\n" + "=" * 55)
    print("      FDPIS FLIGHT DELAY PREDICTION RESULT")
    print("=" * 55)
    print(f" Flight:       {args.carrier.upper()} | {args.origin.upper()} -> {args.dest.upper()}")
    print(f" Schedule:     Dep {args.dep_time:04d} | Arr {args.arr_time:04d} | Day {args.day_of_month}")
    print(f" Aircraft:     {args.tail_num.upper()}")
    print("-" * 55)
    print(f" Delay Probability:        {prob * 100:.2f}%")
    print(f" Calibrated Threshold:     {predictor.optimal_threshold:.2f}")
    if pred == 1:
        print(f" Prediction Status:        DELAYED (>= 15 minutes)")
        print(f" Risk Assessment:          HIGH CASCADE RISK")
    else:
        print(f" Prediction Status:        ON-TIME (< 15 minutes)")
        print(f" Risk Assessment:          LOW CASCADE RISK")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
