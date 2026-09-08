import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

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


# --- Word-building settings you can tune ---
STABILITY_FRAMES = 15   # how many consecutive matching frames = "held steady"
CONFIDENCE_MIN = 0.6    # ignore predictions the model itself isn't confident about

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Live spelling started.")
print("Hold a letter steady to type it. SPACE = space, BACKSPACE = delete, C = clear, ESC = quit.")

start_time = time.time()
typed_text = ""

streak_letter = None
streak_count = 0
locked = False  # True once the current streak has already been typed

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

                features = normalize_landmarks(hand).reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                best_index = np.argmax(probabilities)
                confidence = probabilities[best_index]
                predicted = model.classes_[best_index]

                if confidence >= CONFIDENCE_MIN:
                    current_letter = predicted

        # --- Track how long the same letter has been held ---
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

        # Commit the letter once it's been held steady long enough
        if streak_count >= STABILITY_FRAMES and not locked:
            typed_text += streak_letter
            locked = True  # won't type again until you change shape or drop your hand

        # --- Draw the UI ---
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

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            typed_text += " "
        elif key == 8:  # BACKSPACE
            typed_text = typed_text[:-1]
        elif key in (ord('c'), ord('C')):
            typed_text = ""

cap.release()
cv2.destroyAllWindows()
print("Final typed text:", typed_text)