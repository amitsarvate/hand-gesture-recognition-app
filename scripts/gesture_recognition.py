import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load trained model
model = load_model("models/gesture_recognition_model.h5")

def classify_gesture(hand_landmarks):
    landmarks = np.array(hand_landmarks).flatten()
    landmarks = landmarks - landmarks.min()
    landmarks = landmarks / landmarks.max()
    landmarks = landmarks.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(landmarks)
    gesture_label = np.argmax(prediction)  # Get the class index with the highest probability
    return gesture_label

# Mapping the labels (0-25) to the letters A-Z
gesture_names = {i: chr(65 + i) for i in range(26)}  # 0 -> A, 1 -> B, ..., 25 -> Z

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            gesture_label = classify_gesture(landmarks)
            gesture_name = gesture_names.get(gesture_label, "Unknown Gesture")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
