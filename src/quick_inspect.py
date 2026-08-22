with open(r"C:\Users\ashis\Downloads\T_ONTIME_REPORTING_20260820_081428\T_ONTIME_REPORTING.csv", "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline().strip()
    cols = [c.strip('"') for c in header.split(",")]
    row_count = sum(1 for _ in f)

print(f"Total Rows: {row_count:,}")
print(f"Total Columns: {len(cols)}")
print(f"Columns: {cols}")
