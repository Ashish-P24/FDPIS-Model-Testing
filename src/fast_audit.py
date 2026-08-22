import csv
from collections import Counter

csv_path = r"C:\Users\ashis\Downloads\T_ONTIME_REPORTING_20260820_081428\T_ONTIME_REPORTING.csv"

with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    
    dates = set()
    carriers = Counter()
    origins = Counter()
    dests = Counter()
    tails = Counter()
    dep_del15_counts = Counter()
    cancelled_counts = Counter()
    diverted_counts = Counter()
    
    total = 0
    valid_dep_del15 = Counter()
    
    for row in reader:
        total += 1
        dates.add(row.get("FL_DATE", ""))
        carriers[row.get("OP_UNIQUE_CARRIER", "")] += 1
        origins[row.get("ORIGIN", "")] += 1
        dests[row.get("DEST", "")] += 1
        tails[row.get("TAIL_NUM", "")] += 1
        
        dep_del15 = row.get("DEP_DEL15", "").strip()
        cancelled = row.get("CANCELLED", "").strip()
        diverted = row.get("DIVERTED", "").strip()
        
        dep_del15_counts[dep_del15] += 1
        cancelled_counts[cancelled] += 1
        diverted_counts[diverted] += 1
        
        if cancelled == "0.00" and diverted == "0.00" and dep_del15 != "":
            valid_dep_del15[dep_del15] += 1

print(f"Total rows: {total}")
print(f"Date range: min={min(dates)}, max={max(dates)}, distinct_dates={len(dates)}")
print(f"Unique carriers: {len(carriers)}")
print(f"Top carriers: {carriers.most_common(5)}")
print(f"Unique origins: {len(origins)}, Unique dests: {len(dests)}")
print(f"Unique tail numbers: {len(tails)}")
print(f"Cancelled breakdown: {cancelled_counts}")
print(f"Diverted breakdown: {diverted_counts}")
print(f"DEP_DEL15 all rows: {dep_del15_counts}")
print(f"DEP_DEL15 valid flights (non-cancelled, non-diverted): {valid_dep_del15}")
total_valid = sum(valid_dep_del15.values())
for k, v in valid_dep_del15.items():
    print(f"  Class {k}: {v:,} ({v/total_valid*100:.2f}%)")
