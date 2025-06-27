from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import joblib
import base64
from flask_cors import CORS
import time
from collections import deque, Counter

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load("bsl_model.pkl")
scaler = joblib.load("scaler.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

# --- Word accumulation state ---
current_word = ""
last_prediction = None
last_added_time = 0
COOLDOWN_SECONDS = 0.3  # Adjust as needed
SMOOTHING_WINDOW = 5  # You can try 3, 5, or 7
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# -- New state for the Prediction Stability system --
STABILITY_THRESHOLD = 3  # Number of consecutive frames for a prediction to be "stable"
stable_candidate = None
stable_candidate_count = 0

def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks_list = []
    all_x, all_y = [], []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])
                h, w, _ = image.shape
                all_x.append(int(lm.x * w))
                all_y.append(int(lm.y * h))
        # Pad for single hand
        if len(landmarks_list) == 63:
            landmarks_list += [0.0] * 63
        elif len(landmarks_list) != 126:
            return None, None, None, None
        return landmarks_list, all_x, all_y, results
    return None, None, None, None

@app.route('/predict_word', methods=['POST'])
def predict_word():
    global current_word, last_prediction, last_added_time, stable_candidate, stable_candidate_count
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    image_data = data['image'].split(',')[1]  # Remove data:image/png;base64,
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    landmarks, all_x, all_y, results = extract_landmarks(img)
    display_pred = None

    if not landmarks:
        # Always return the original image as base64 if no hand detected
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return jsonify({'prediction': None, 'word': current_word, 'error': 'No hand detected', 'image': img_base64})

    X = np.array(landmarks).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(proba)
    pred_confidence = proba[pred_idx]
    prediction = model.classes_[pred_idx]

    # Clean prediction: show only number or letter
    parts = str(prediction).split('-')
    if len(parts) == 2 and parts[1].strip().isdigit():
        display_pred = parts[1].strip()  # Show number only
    else:
        display_pred = parts[-1].strip()  # Show letter only

    # --- Smoothing: Majority vote over last N predictions ---
    prediction_history.append(display_pred)
    if len(prediction_history) == SMOOTHING_WINDOW:
        most_common_pred, count = Counter(prediction_history).most_common(1)[0]
    else:
        most_common_pred = display_pred

    # --- Prediction Stability Logic ---
    if most_common_pred == stable_candidate:
        stable_candidate_count += 1
    else:
        # If the prediction changes, reset the stability counter
        stable_candidate = most_common_pred
        stable_candidate_count = 1

    # A prediction is only "confirmed" if it has been stable for N frames
    confirmed_prediction = None
    if stable_candidate_count >= STABILITY_THRESHOLD:
        confirmed_prediction = stable_candidate

    # Draw bounding box if hand detected (shows the smoothed prediction for instant feedback)
    if all_x and all_y:
        x1, y1 = min(all_x), min(all_y)
        x2, y2 = max(all_x), max(all_y)
        cv2.rectangle(img, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)
        cv2.putText(img, f"{most_common_pred}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # --- Accumulate word logic (using the STABLE prediction) ---
    CONFIDENCE_THRESHOLD = 0.3
    current_time = time.time()
    if (
        confirmed_prediction is not None
        and pred_confidence >= CONFIDENCE_THRESHOLD
        and confirmed_prediction != last_prediction
    ):
        current_word += str(confirmed_prediction)
        last_prediction = confirmed_prediction # Update the last character that was officially added
        last_added_time = current_time

    # Encode processed image to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return jsonify({
        'prediction': most_common_pred, # Show the user what is currently being seen for feedback
        'confidence': float(pred_confidence),
        'word': current_word, # This will only update when a prediction is stable and confirmed
        'image': img_base64
    })

@app.route('/finish_word', methods=['POST'])
def finish_word():
    global current_word
    finished = current_word
    current_word = ""
    return jsonify({'finished_word': finished})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
