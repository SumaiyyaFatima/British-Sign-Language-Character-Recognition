import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd

# MediaPipe setup
mp_hands = mp.solutions.hands  # type: ignore
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

# Webcam setup - use index 1 for external webcam
cap = cv2.VideoCapture(0)
# Dataset file
DATASET_FILE = "BSL_Dataset.csv"

# Signs to collect (you can modify this list)
SIGNS_TO_COLLECT = [
    "W-w"
]

def extract_landmarks(image):
    """Extract hand landmarks from image"""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])
        # Pad for single hand
        if len(landmarks_list) == 63:
            landmarks_list += [0.0] * 63
        elif len(landmarks_list) != 126:
            return None
        return landmarks_list
    return None

def save_landmarks_to_csv(landmarks, label):
    """Save landmarks and label to CSV file"""
    if not os.path.exists(DATASET_FILE):
        header = [f"landmark_{i}" for i in range(126)] + ["label"]
        with open(DATASET_FILE, "w") as f:
            f.write(",".join(header) + "\n")
    
    with open(DATASET_FILE, "a") as f:
        landmark_str = ",".join(map(str, landmarks))
        f.write(f"{landmark_str},{label}\n")

def collect_data_for_sign(sign_label, samples_per_sign=10, countdown_time=3):
    """Collect multiple samples for a specific sign using a timer."""
    print(f"\n=== Collecting data for: {sign_label} ===")
    print(f"Get ready to make the sign for '{sign_label}' {samples_per_sign} times.")
    print(f"An image will be captured automatically after a {countdown_time}-second countdown.")
    print("Press 'q' to quit, 's' to skip this sign.")
    
    samples_collected = 0
    
    while samples_collected < samples_per_sign:
        print(f"\nCapturing sample {samples_collected + 1}/{samples_per_sign} for {sign_label}...")
        
        # Start countdown
        for i in range(countdown_time, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            # Display instructions
            cv2.putText(frame, f"Sign: {sign_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing in: {i}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)  # Wait 1 second
            
        # Capture the frame after countdown
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        landmarks = extract_landmarks(frame)
        if landmarks:
            save_landmarks_to_csv(landmarks, sign_label)
            samples_collected += 1
            print(f"Captured sample {samples_collected}/{samples_per_sign} for {sign_label}")
            
            # Show a "Captured!" message
            cv2.putText(frame, "Captured!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000) # Show for 1 second
        else:
            print("No hand detected! Please show your hand clearly. Retrying...")
            cv2.putText(frame, "No hand detected. Retrying...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(2000) # Wait 2 seconds before retry

        # Check for quit/skip keys during the process
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit completely
        elif key == ord('s'):
            print(f"Skipping {sign_label}")
            return True  # Continue to next sign
            
    print(f"Completed collecting {samples_collected} samples for {sign_label}")
    return True

def main():
    print("=== BSL Data Collection Tool ===")
    print("This tool will help you collect new training data for your BSL model.")
    print("Make sure you have good lighting and a plain background.")
    
    if os.path.exists(DATASET_FILE):
        print(f"Found existing dataset: {DATASET_FILE}")
        # Count existing samples, ignoring comment lines and skipping bad lines
        try:
            df = pd.read_csv(DATASET_FILE, comment='#', on_bad_lines='skip')
            print(f"Current dataset has {len(df)} samples")
        except Exception as e:
            print(f"Could not read dataset file: {e}")
            return
    else:
        print("Creating new dataset file")
    
    print(f"\nWill collect data for {len(SIGNS_TO_COLLECT)} signs")
    
    try:
        samples_per_sign = int(input("How many samples per sign? (default: 10): ") or "10")
    except ValueError:
        samples_per_sign = 10
    
    print("\nStarting data collection...")
    print("A photo will be taken automatically after a countdown.")
    print("Press 's' to skip a sign, 'q' to quit early.")
    
    for sign in SIGNS_TO_COLLECT:
        if not collect_data_for_sign(sign, samples_per_sign):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Data Collection Complete ===")
    print("You can now retrain your model using:")
    print("python train_model.py")

if __name__ == "__main__":
    main() 