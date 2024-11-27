import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mapping keys (A-Z) to ASL letter labels
letter_labels = {i: chr(65 + i) for i in range(26)}  # Create a dictionary: 0 -> A, 1 -> B, ..., 25 -> Z

# Ensure the 'data' directory exists
if not os.path.exists("data"):
    os.makedirs("data")

def log_landmarks(landmarks, label):
    with open("data/gesture_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        row = np.hstack((landmarks.flatten(), label))
        print("Writing row:", row)  # Debugging the row being written
        writer.writerow(row)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_label = None
    key = cv2.waitKey(1) & 0xFF  # Get the key press as an integer

    if key >= ord('a') and key <= ord('z'):  # Check if the key pressed is between 'a' and 'z'
        gesture_label = key - ord('a')  # Convert key to a number (0 for 'a', 1 for 'b', ..., 25 for 'z')
        print(f"Collecting data for: {letter_labels[gesture_label]}")  # Display the letter being collected

    if results.multi_hand_landmarks and gesture_label is not None:
        print(f"Detected {len(results.multi_hand_landmarks)} hands")
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            print("Landmarks:", landmarks)  # Debugging landmarks
            log_landmarks(landmarks, gesture_label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    
    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
