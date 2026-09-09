import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
import math
from collections import deque

MODEL_PATH = "hand_landmarker.task"
DATA_FILE = "landmark_data.csv"

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

MOTION_KEYS = {ord('j'): 'j', ord('z'): 'z'}


def normalize_landmarks(hand_landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = points[0].copy()
    points -= wrist
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points.flatten(), scale if scale > 0 else 1.0


def compute_motion_features(fingertip_path):
    """fingertip_path: list of (x, y) normalized index-fingertip positions
    over the recording window."""
    if len(fingertip_path) < 2:
        return [0, 0, 0, 0, 0, 0]

    path_length = 0.0
    direction_changes = 0
    prev_dx, prev_dy = None, None

    xs = [p[0] for p in fingertip_path]
    ys = [p[1] for p in fingertip_path]

    for i in range(1, len(fingertip_path)):
        x1, y1 = fingertip_path[i - 1]
        x2, y2 = fingertip_path[i]
        dx, dy = x2 - x1, y2 - y1
        path_length += math.hypot(dx, dy)

        if prev_dx is not None:
            # Count a "direction change" if the movement's dominant axis flips sign
            if (dx * prev_dx < 0) or (dy * prev_dy < 0):
                direction_changes += 1
        prev_dx, prev_dy = dx, dy

    start = fingertip_path[0]
    end = fingertip_path[-1]
    net_displacement = math.hypot(end[0] - start[0], end[1] - start[1])

    straightness = net_displacement / path_length if path_length > 0 else 0

    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    return [path_length, net_displacement, straightness, bbox_width, bbox_height, direction_changes]


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Hold 'j' or 'z' down while performing the motion, then release.")
print("Press ESC to quit.")
start_time = time.time()

recording_letter = None
landmark_smoothing = deque(maxlen=4)
recorded_frames = []  # list of (normalized_landmarks_63, fingertip_xy)

counts = {"j": 0, "z": 0}

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
        current_normalized = None
        current_fingertip = None

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                raw_normalized, _ = normalize_landmarks(hand)
                landmark_smoothing.append(raw_normalized)

                # Average the last few frames together to smooth out jitter/glitches
                # from moments where the camera struggles to see the hand clearly
                current_normalized = np.mean(landmark_smoothing, axis=0)
                current_fingertip = (current_normalized[8 * 3], current_normalized[8 * 3 + 1])

        # If currently recording, keep appending frames
        if recording_letter is not None and current_normalized is not None:
            recorded_frames.append((current_normalized, current_fingertip))
            cv2.putText(frame, f"RECORDING {recording_letter.upper()}... ({len(recorded_frames)} frames)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        counts_text = f"J: {counts['j']}   Z: {counts['z']}"
        cv2.putText(frame, counts_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, "Hold J or Z key through the motion. ESC to quit.",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Collect Motion Data (J/Z)", frame)

        cv2.imshow("Collect Motion Data (J/Z)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key in MOTION_KEYS:
            letter = MOTION_KEYS[key]

            if recording_letter is None:
                # Not currently recording -- this press STARTS a recording
                recording_letter = letter
                recorded_frames = []
                print(f"Started recording {letter.upper()}...")

            elif recording_letter == letter:
                # Already recording this same letter -- this press STOPS and saves it
                if len(recorded_frames) >= 10:
                    last_normalized = recorded_frames[-1][0]
                    fingertip_path = [f[1] for f in recorded_frames]
                    motion_features = compute_motion_features(fingertip_path)

                    row = [recording_letter] + list(last_normalized) + motion_features
                    with open(DATA_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    counts[recording_letter] += 1
                    print(f"Saved a {recording_letter.upper()} sample ({len(recorded_frames)} frames)")
                else:
                    print("Too short, discarded. Try a slightly longer motion.")

                recording_letter = None
                recorded_frames = []    
            
cap.release()
cv2.destroyAllWindows()
print("Session ended.")
print(f"J samples: {counts['j']}, Z samples: {counts['z']}")