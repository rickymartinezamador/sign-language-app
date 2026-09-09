import csv
import os

DATA_FILE = "landmark_data.csv"
BACKUP_FILE = "landmark_data_backup.csv"

MOTION_COLUMNS = [
    "path_length", "net_displacement", "straightness",
    "bbox_width", "bbox_height", "direction_changes"
]

# Make a backup first, just in case
if not os.path.exists(BACKUP_FILE):
    with open(DATA_FILE, "r", newline="") as src, open(BACKUP_FILE, "w", newline="") as dst:
        dst.write(src.read())
    print(f"Backed up original data to {BACKUP_FILE}")

rows = []
with open(DATA_FILE, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)

if MOTION_COLUMNS[0] in header:
    print("Already migrated — no changes made.")
else:
    new_header = header + MOTION_COLUMNS
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        for row in rows:
            writer.writerow(row + [0, 0, 0, 0, 0, 0])
    print(f"Migrated {len(rows)} rows. New columns: {MOTION_COLUMNS}")