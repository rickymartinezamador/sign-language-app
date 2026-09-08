import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "landmark_data.csv"
MODEL_OUT = "sign_model.pkl"


def normalize_landmarks(flat_coords):
    """Make coordinates relative to the wrist (landmark 0) and scale
    by hand size, so position/distance from camera don't matter."""
    points = np.array(flat_coords).reshape(21, 3)
    wrist = points[0].copy()
    points -= wrist  # shift so wrist is the origin

    # scale by the distance from wrist to middle-finger MCP (landmark 9),
    # a stable reference for "how big is this hand in frame"
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale

    return points.flatten()


# Load raw data
labels = []
features = []

with open(DATA_FILE, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        label = row[0]
        coords = [float(v) for v in row[1:]]
        features.append(normalize_landmarks(coords))
        labels.append(label)

X = np.array(features)
y = np.array(labels)

print(f"Loaded {len(y)} samples across {len(set(y))} letters.")

# Split into training and test sets so we can honestly check accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy: {accuracy:.2%}\n")
print(classification_report(y_test, predictions))

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_OUT}")