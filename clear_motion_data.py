import csv

DATA_FILE = "landmark_data.csv"

rows = []
with open(DATA_FILE, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)

kept_rows = [row for row in rows if row[0] not in ("j", "z")]
removed_count = len(rows) - len(kept_rows)

with open(DATA_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(kept_rows)

print(f"Removed {removed_count} J/Z rows. {len(kept_rows)} rows remain.")