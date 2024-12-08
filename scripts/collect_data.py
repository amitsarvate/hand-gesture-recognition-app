# --------------------------------------------------------------------------------------------
#
# Title: Collect Data (Project Part 1)
# Course: CSCI 3240 (Computational Photography)
# Authors: Amit Sarvate and Nirujan Velvarathan
# Date: December 8th 2024
#
# Description: The purpose of this script is to allow us to collect the training data that we 
# will feed to our classification model in order to enable real-time hand gesture recognition 
#
# --------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import csv

#  Initialize MediaPipe Hands library module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Applying a sharpening filter to the frame 
def sharpen_image(frame):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]) 
    return cv2.filter2D(frame, -1, kernel) # Convolving with our signal 

# Logging hand landmark coordinates along with gesture label assigned into a CSV 
def log_landmarks(landmarks, label):
    with open("data/gesture_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(np.hstack((landmarks.flatten(), label)))

# Initializing MediaPipe hands 
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capturing video from laptop's webcam 
cap = cv2.VideoCapture(0)

while True:

    # Reading frame from webcam 
    ret, frame = cap.read()
    if not ret:
        break

    # Flippinh frame using cv2.flip to create mirroe view 
    frame = cv2.flip(frame, 1)

    # Apply sharpening kernel to signal (frame)
    frame = sharpen_image(frame)

    # Converting frame from BGR to RGB colour space 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing frame to detect hand and landmarks on hand from frame 
    results = hands.process(rgb_frame)

    # Depending on the gesture currently being recorded the user will need to press a specific key ... 
    # ... This will give each instanance of landmarks an associated class label (used for our classification model)
    gesture_label = None
    if cv2.waitKey(1) & 0xFF == ord('1'):
        gesture_label = 1  # Label for "Thumbs Up"
    elif cv2.waitKey(1) & 0xFF == ord('2'):
        gesture_label = 2  # Label for "OK Sign"
    elif cv2.waitKey(1) & 0xFF == ord('3'):
        gesture_label = 3  # Label for "Victory"
    elif cv2.waitKey(1) & 0xFF == ord('4'):
        gesture_label = 4  # Label for "Fist"
    elif cv2.waitKey(1) & 0xFF == ord('5'):
        gesture_label = 5  # Label for "Stop Sign"

    # If hand landmarks are detected and a gesture label has been assigned
    if results.multi_hand_landmarks and gesture_label:
        for hand_landmarks in results.multi_hand_landmarks:

            # Extract x,y,z coordinates from each landmark 
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

            # Use function defined above in order to log the landmarks and gesture label into the CSV file 
            log_landmarks(landmarks, gesture_label)

            # Draw landmarks and connection onto the frame (helps with visualization)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Data Collection", frame)

    # Exit the loop if the ESC key is pressed 
    if cv2.waitKey(1) & 0xFF == 27:  
        break

# Replease the webcam and close OpenCV windows 
cap.release()
cv2.destroyAllWindows()
