import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math
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
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = points[0].copy()
    points -= wrist
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points.flatten()


def compute_motion_features(fingertip_path):
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


STABILITY_FRAMES = 15
CONFIDENCE_MIN = 0.6
MOTION_WINDOW = 20      # how many recent frames of fingertip position we track
SMOOTHING_WINDOW = 4    # how many frames we average together to reduce jitter

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Live spelling started (now with J and Z!).")
print("Hold a letter steady, or perform J/Z's motion, to type it.")
print("SPACE = space, BACKSPACE = delete, C = clear, ESC = quit.")

start_time = time.time()
typed_text = ""

streak_letter = None
streak_count = 0
locked = False

landmark_smoothing = deque(maxlen=SMOOTHING_WINDOW)
fingertip_history = deque(maxlen=MOTION_WINDOW)

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
        current_letter = None

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                raw_normalized = normalize_landmarks(hand)
                landmark_smoothing.append(raw_normalized)
                smoothed = np.mean(landmark_smoothing, axis=0)

                fingertip_xy = (smoothed[8 * 3], smoothed[8 * 3 + 1])
                fingertip_history.append(fingertip_xy)

                motion_features = compute_motion_features(list(fingertip_history))

                features = np.concatenate([smoothed, motion_features]).reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                best_index = np.argmax(probabilities)
                confidence = probabilities[best_index]
                predicted = model.classes_[best_index]

                if confidence >= CONFIDENCE_MIN:
                    current_letter = predicted
        else:
            # Hand left the frame -- clear the motion history so old
            # movement doesn't bleed into the next gesture
            landmark_smoothing.clear()
            fingertip_history.clear()

        if current_letter is not None and current_letter == streak_letter:
            streak_count += 1
        elif current_letter is not None:
            streak_letter = current_letter
            streak_count = 1
            locked = False
        else:
            streak_letter = None
            streak_count = 0
            locked = False

        if streak_count >= STABILITY_FRAMES and not locked:
            typed_text += streak_letter
            locked = True

        if current_letter:
            cv2.putText(frame, current_letter.upper(), (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 5)

            progress = min(streak_count / STABILITY_FRAMES, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (30, 100), (230, 120), (100, 100, 100), 2)
            cv2.rectangle(frame, (30, 100), (30 + bar_width, 120), (0, 255, 0), -1)

        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, typed_text[-40:], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, "SPACE=space  BACKSPACE=delete  C=clear  ESC=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Language Spelling", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            typed_text += " "
        elif key == 8:
            typed_text = typed_text[:-1]
        elif key in (ord('c'), ord('C')):
            typed_text = ""

cap.release()
cv2.destroyAllWindows()
print("Final typed text:", typed_text)