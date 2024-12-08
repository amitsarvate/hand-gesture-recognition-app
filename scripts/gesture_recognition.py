# --------------------------------------------------------------------------------------------
#
# Title: Gesture Recognition based Image Manipulation (Project Part 3)
# Course: CSCI 3240 (Computational Photography)
# Authors: Amit Sarvate and Nirujan Velvarathan
# Date: December 8th 2024
#
# Description: The purpose of this script is to detect hand gestures in real time and, based on 
# those hand gestures, manipulate an image in real time.
#
# --------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import PySimpleGUI as sg
from PIL import Image, ImageTk

# Load trained model - trained in Part 2 using data collected from Part 1 
model = load_model("models/gesture_recognition_model.h5")

# Gesture to action mapping 
gesture_actions = {
    0: "Increase Brightness",
    1: "Grayscale",
    2: "Edge Detection",
    3: "Reset Image",
    4: "Invert Colors"
}

# Classifying gestures using the current hand larkmarks and the model to predict classification 
def classify_gesture(hand_landmarks):
    landmarks = np.array(hand_landmarks).flatten()
    landmarks = landmarks - landmarks.min()
    landmarks = landmarks / landmarks.max()
    landmarks = landmarks.reshape(1, -1)  
    prediction = model.predict(landmarks)
    gesture_label = np.argmax(prediction)
    return gesture_label

# Increases brightness of the image 
def increase_brightness(image):
    return cv2.convertScaleAbs(image, alpha=1.2, beta=30)

# Applies a greyscale to the image 
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applies edge detection to the image 
def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

# Inverts colours of the image 
def invert_colors(image):
    return cv2.bitwise_not(image)

# Initialize MediaPipe Hands to track hand movements
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load and preprocess the image that will be manipulated 
image_path = "imgs/IMG_1329.jpeg" 
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
current_image = original_image.copy()

# Defining constants for dimension of the display 
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Defining our layout 
layout = [
    [sg.Text("Real-Time Gesture Recognition", size=(30, 1), justification="center"),
     sg.Text("Image Manipulation", size=(30, 1), justification="center")],
    [sg.Image(filename="", key="video_feed", size=(DISPLAY_WIDTH, DISPLAY_HEIGHT)),
     sg.Image(filename="", key="image_display", size=(DISPLAY_WIDTH, DISPLAY_HEIGHT))],
    [sg.Text("Gesture: None", key="gesture_label", size=(50, 1))]
]

window = sg.Window("Gesture-Based Image Manipulation", layout, location=(100, 100), resizable=True)

# Capturing feed from laptop's webcam 
cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=10)
    if event == sg.WINDOW_CLOSED:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(rgb_frame)

    gesture_name = "None"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Extracting hand landmarks and classifiying gesture 
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            gesture_label = classify_gesture(landmarks)
            gesture_name = gesture_actions.get(gesture_label, "Unknown Gesture")

            # Perform image manipulation based on gesture
            if gesture_name == "Increase Brightness":
                current_image = increase_brightness(original_image)
            elif gesture_name == "Grayscale":
                current_image = apply_grayscale(original_image)
            elif gesture_name == "Edge Detection":
                current_image = apply_edge_detection(original_image)
            elif gesture_name == "Reset Image":
                current_image = original_image.copy()
            elif gesture_name == "Invert Colors":
                current_image = invert_colors(original_image)

            # Draw landmarks on the video feed
            mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Updating video feed and resizing 
    frame_resized = cv2.resize(rgb_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR) 
    imgbytes = cv2.imencode(".png", frame_bgr)[1].tobytes()
    window["video_feed"].update(data=imgbytes)

    # Resize and update the manipulated image display
    if len(current_image.shape) == 2:  
        current_image_display = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    else:
        current_image_display = current_image
    current_image_display = cv2.resize(current_image_display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    img = Image.fromarray(current_image_display)
    imgbytes = ImageTk.PhotoImage(img)
    window["image_display"].update(data=imgbytes)

    # Update gesture label
    window["gesture_label"].update(f"Gesture: {gesture_name}")

cap.release()
cv2.destroyAllWindows()
window.close()
