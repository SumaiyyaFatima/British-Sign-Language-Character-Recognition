import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter
import time

# Load model and scaler
model = joblib.load("bsl_model.pkl")
scaler = joblib.load("scaler.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands  # type: ignore
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils  # type: ignore

# Webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")

# === Word Accumulation Setup ===
current_word = ""
last_prediction = None
word_display_time = 0
finished_word = ""

# === Smoothing Setup ===
N_SMOOTH = 3
prediction_history = deque(maxlen=N_SMOOTH)

# === Start/Pause Control ===
predicting = False
countdown_done = False

CONFIDENCE_THRESHOLD = 0.5  # Stable threshold
COOLDOWN_SECONDS = 0.1
last_added_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if not predicting:
        cv2.putText(frame, "Press 's' to start prediction", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.imshow("BSL Real-Time", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Start countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Starting in {i}...", (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
                cv2.imshow("BSL Real-Time", frame)
                cv2.waitKey(1000)
            predicting = True
            countdown_done = True
        continue

    # Only run prediction after countdown
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks_list = []
    all_x, all_y = [], []

    prediction = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])
                h, w, _ = frame.shape
                all_x.append(int(lm.x * w))
                all_y.append(int(lm.y * h))

        # === Padding for single hand ===
        if len(landmarks_list) == 63:
            landmarks_list += [0.0] * 63
        elif len(landmarks_list) != 126:
            cv2.putText(frame, "Landmark mismatch", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("BSL Real-Time", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # === Predict ===
        X = np.array(landmarks_list).reshape(1, -1)
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_confidence = proba[pred_idx]
        prediction = model.classes_[pred_idx]
        prediction_history.append(prediction)

        # === Smoothing: Use majority vote ===
        if len(prediction_history) == N_SMOOTH:
            most_common_pred, count = Counter(prediction_history).most_common(1)[0]
        else:
            most_common_pred = prediction

        # === Extract display part based on type ===
        parts = most_common_pred.split('-')
        if len(parts) == 2 and parts[1].isdigit():
            display_pred = parts[0]  # Show word for numbers
        else:
            display_pred = parts[-1]  # Show letter for alphabets

        # === Draw combined bounding box ===
        if all_x and all_y:
            x1, y1 = min(all_x), min(all_y)
            x2, y2 = max(all_x), max(all_y)
            cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)
            if pred_confidence >= CONFIDENCE_THRESHOLD:
                cv2.putText(frame, f"{display_pred}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "...", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # === Accumulate Characters (with smoothing, confidence, and cooldown) ===
        current_time = time.time()
        if (
            display_pred is not None
            and pred_confidence >= CONFIDENCE_THRESHOLD
            and display_pred != last_prediction
            and (current_time - last_added_time) > COOLDOWN_SECONDS
        ):
            current_word += str(display_pred)
            last_prediction = display_pred
            last_added_time = current_time
    else:
        last_prediction = None
        prediction_history.clear()
        cv2.putText(frame, "No hands detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
        cv2.rectangle(frame, (10, 60), (350, 100), (100, 100, 255), -1)
        cv2.putText(frame, "Show a sign to get prediction", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # === Display Current Word ===
    cv2.putText(frame, f"Word: {current_word}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)

    # === Display Finished Word if any ===
    if finished_word and (cv2.getTickCount() - word_display_time < cv2.getTickFrequency() * 2):
        cv2.putText(frame, f"Finished: {finished_word}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    elif finished_word:
        finished_word = ""

    cv2.imshow("BSL Real-Time", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Space bar to finish word
        if current_word:
            finished_word = current_word
            word_display_time = cv2.getTickCount()
            print(f"Word completed: {finished_word}")
            current_word = ""

cap.release()
cv2.destroyAllWindows()
