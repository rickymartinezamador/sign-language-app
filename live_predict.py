import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque, Counter

MODEL_PATH = "hand_landmarker.task"
SIGN_MODEL_PATH = "sign_model.pkl"

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

with open(SIGN_MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def normalize_landmarks(hand_landmarks):
    """Same normalization used during training: relative to the wrist,
    scaled by hand size."""
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = points[0].copy()
    points -= wrist
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points.flatten()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Live prediction started. Press ESC to quit.")
start_time = time.time()

# Keep a short rolling history of predictions to smooth out flicker
recent_predictions = deque(maxlen=10)

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
        predicted_letter = ""

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                features = normalize_landmarks(hand).reshape(1, -1)
                prediction = model.predict(features)[0]
                recent_predictions.append(prediction)

            # Show the most common prediction over the last few frames
            most_common = Counter(recent_predictions).most_common(1)[0][0]
            predicted_letter = most_common
        else:
            recent_predictions.clear()

        if predicted_letter:
            cv2.putText(frame, predicted_letter.upper(), (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

        cv2.putText(frame, "Press ESC to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()