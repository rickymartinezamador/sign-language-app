import cv2
import mediapipe as mp
import urllib.request
import os
import time
import csv

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

DATA_FILE = "landmark_data.csv"
VALID_LETTERS = set("abcdefghiklmnopqrstuvwxy")  # skip j and z (they involve motion)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

counts = {letter: 0 for letter in VALID_LETTERS}
with open(DATA_FILE, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if row and row[0] in counts:
            counts[row[0]] += 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Hold a letter shape, then press that letter key to save a sample.")
print("Press ESC to quit.")
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        current_landmarks = None

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                current_landmarks = hand
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(frame, "Press a letter key to save a sample. ESC to quit.",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset = 55
        sorted_letters = sorted(counts.keys())
        row_text = ""
        for i, letter in enumerate(sorted_letters):
            row_text += f"{letter}:{counts[letter]}  "
            if (i + 1) % 8 == 0:
                cv2.putText(frame, row_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25
                row_text = ""
        if row_text:
            cv2.putText(frame, row_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Collect Sign Language Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        pressed_char = chr(key) if key != 255 else ""
        if pressed_char in VALID_LETTERS and current_landmarks is not None:
            row = [pressed_char]
            for lm in current_landmarks:
                row += [lm.x, lm.y, lm.z]
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            counts[pressed_char] += 1

cap.release()
cv2.destroyAllWindows()
print("Session ended. Samples saved to", DATA_FILE)
for letter in sorted(counts.keys()):
    print(f"  {letter}: {counts[letter]}")