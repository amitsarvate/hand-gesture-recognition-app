import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def log_landmarks(landmarks, label):
    with open("data/gesture_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(np.hstack((landmarks.flatten(), label)))

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

key_to_gesture_label = {
        '1': 1,  # Label for "Thumbs Up"
        '2': 2,  # Label for "OK Sign"
        '3': 3,  # Label for "Victory"
        '4': 4,  # Label for "Fist"
        '5': 5,  # Label for "Stop Sign",
        **{chr(i): i - 91 for i in range(ord('a'), ord('z') + 1)}  # Map 'a' to 'z' to labels 6-31
    }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


    gesture_label = None

    # Capture key press
    key = cv2.waitKey(1) & 0xFF  # Get the key code
    if key != -1:  # Ensure a key was pressed
        key_char = chr(key)  # Convert key code to character
        gesture_label = key_to_gesture_label.get(key_char, None)  # Get gesture label from dictionary

    # Use the gesture_label variable to perform further actions
    if gesture_label is not None:
        print(f"Gesture detected: {gesture_label}")


    if results.multi_hand_landmarks and gesture_label:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            log_landmarks(landmarks, gesture_label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
